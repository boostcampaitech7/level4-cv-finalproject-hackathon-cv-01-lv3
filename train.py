import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model.sources.model_config import VideoChat2Config
from model.sources.modeling_videochat2 import InternVideo2_VideoChat2
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from model.utils.data_utils_from_json import InternVideo2_VideoChat2_Dataset, InternVideo2_VideoChat2_DataLoader
from tqdm import tqdm
# BERTScore 계산을 위함 (사용 시, pip install bert_score 이후, 아래 1줄 주석 해제)
# from bert_score import score
import wandb
import json
from datetime import datetime

def train(
    model_path: str,
    data_path: str = "../../data",
    num_epochs: int = 50,
    train_batch_size: int = 2,
    test_batch_size: int = 1,
    train_num_workers: int = 4,
    test_num_workers: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    validation_interval: int = 5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    모델을 학습시키는 함수, 일정 주기로 validation을 수행함.

    Args: 
        model_path: 모델 경로
        data_path: 데이터 경로
        num_epochs: 학습 주기
        train_batch_size: 학습 배치 크기
        test_batch_size: 검증 배치 크기
        train_num_workers: 학습 데이터 로더 스레드 수
        test_num_workers: 검증 데이터 로더 스레드 수
        learning_rate: 학습률
        weight_decay: 가중치 감쇠 비율
        validation_interval: 검증 주기
        device: CPU or GPU
    """

    # 로깅을 위한 디렉토리 생성
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)

    # 기본 설정 로깅
    config = {
        "model_path": model_path,
        "data_path": data_path,
        "num_epochs": num_epochs,
        "train_batch_size": train_batch_size,
        "test_batch_size": test_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "device": device,
        "system": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        }
    }
    
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # wandb 초기화
    wandb.init(
        project="videochat2-training",
        config=config
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = VideoChat2Config.from_json_file(
        os.path.join(current_dir,'model','configs', 'config.json')
    )

    # 토크나이저 초기화 (Mistral-7B)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_config.llm.pretrained_llm_path,
        trust_remote_code=True,
        use_fast=False,
        token=os.getenv('HF_TOKEN')
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # 모델 초기화
    model = InternVideo2_VideoChat2.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)

    # 모델 내부적으로 Attention, Loss, Output 출력 등에서 pad_token이 불필요한 영향을 주지 않도록 설정
    model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = InternVideo2_VideoChat2_Dataset(
        data_path=data_path,
        use_segment=True,
        use_audio=False,
        train=True
    )

    test_dataset = InternVideo2_VideoChat2_Dataset(
        data_path=data_path,
        use_segment=True,
        use_audio=False,
        train=False
    )
    
    train_loader = InternVideo2_VideoChat2_DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=train_num_workers,
        shuffle=True,
        pin_memory=True,
        use_audio=False
    )
    
    test_loader = InternVideo2_VideoChat2_DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=test_num_workers,
        shuffle=False,
        pin_memory=True,
        use_audio=False
    )
    
    optimizer, scheduler = model.prepare_for_training(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    query_embedding_size = model.query_tokens.shape[1] + model.extra_query_tokens.shape[1]

    # 모델 정보 로깅
    model_info = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_structure": str(model)
    }
    
    with open(os.path.join(log_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=4)

    # 학습 로그 파일 생성
    log_file = os.path.join(log_dir, 'training.log')
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")

    for epoch in range(num_epochs):
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loop):

            frames = batch['frames'].to(device)
            annotations = batch['annotations']
            
            # 텍스트 토큰화
            text_inputs = tokenizer(
                annotations,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)

            optimizer.zero_grad()
            outputs, _ = model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                video=frames,
                labels=text_inputs.input_ids,
                video_idx=torch.ones(frames.shape[0], query_embedding_size).to(device)
            )

            loss = outputs.loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loop.set_postfix(loss=f'{loss.item():.4f}, batch_idx: {batch_idx}')
            
            del frames, annotations, text_inputs, outputs, loss

            # 에폭 정보 로깅
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}\n")

        # 에폭별 평균 손실 로깅
        avg_epoch_loss = epoch_loss / len(train_loader)
        wandb.log({
            "epoch": epoch,
            "epoch_loss": avg_epoch_loss
        })

        if epoch % validation_interval == 0:
            print("--------------------------------")
            print(f"validation start, epoch: {epoch+1}")
            # 모델 저장 방식: (임시) 마지막 Checkpoint를 기준으로 1개만 저장되도록 유지됨
            val_loss = validation(model, test_loader, tokenizer, device, query_embedding_size)
            
            # validation 결과 로깅 
            wandb.log({
                "epoch": epoch,
                "validation_loss": val_loss
            })
            
            save_model(model, optimizer=optimizer, epoch=epoch, loss=None, save_path=os.path.join('temp_model', 'best_model.pt'))
            print(f"validation end, epoch: {epoch+1}")
            print("--------------------------------")
            model.train()

            with open(log_file, 'a') as f:
                f.write(f"Validation - Epoch {epoch}: Loss = {val_loss:.4f}\n")
                f.write("-" * 50 + "\n")

    wandb.finish()

def save_model(model, optimizer=None, scheduler=None, epoch=None, loss=None, save_path="best_model.pt"):
    """
    모델의 파라미터와 함께, optimizer, scheduler, epoch, loss를 저장하는 함수입니다

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer, optional): Optimizer state to save. Default is None.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler state to save. Default is None.
        epoch (int, optional): Current epoch. Default is None.
        loss (float, optional): Best validation loss. Default is None.
        save_path (str, optional): File path to save the model. Default is "best_model.pt".
    """
    # Ensure the save directory exists
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir) and save_dir != "":
        os.makedirs(save_dir)
    
    # Prepare the state dictionary
    state = {
        'model_state_dict': model.state_dict(),
    }
    if optimizer:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler:
        state['scheduler_state_dict'] = scheduler.state_dict()
    if epoch is not None:
        state['epoch'] = epoch
    if loss is not None:
        state['best_loss'] = loss
    
    # Save the state dictionary
    torch.save(state, save_path)
    print(f"Model saved to {save_path}")

def validation(model, dataloader, tokenizer, device, query_embedding_size):
    model.eval()
    total_score = 0
    val_loop = tqdm(dataloader, desc='Validation')
    with torch.no_grad():
        for batch in val_loop:
            frames = batch['frames'].to(device)
            annotations = batch['annotations']
            
            # 이후 평가 메트릭 도입하여 모델 저장 필요
            # text_inputs = tokenizer(
            #     annotations,
            #     padding='longest',
            #     truncation=True,
            #     max_length=256,
            #     return_tensors="pt"
            # ).to(device)

            # _, text_embeds = model(
            #     input_ids=text_inputs.input_ids,
            #     attention_mask=text_inputs.attention_mask,
            #     video=frames,
            #     labels=text_inputs.input_ids,
            #     video_idx=torch.ones(frames.shape[0], query_embedding_size).to(device)
            # )
            
            generation_config = {
                'num_beams': 1,            # 빔 서치 크기
                'max_new_tokens': 200,     # 최대 생성 길이
                'do_sample': False,         # 샘플링 사용
                'top_p': 0.9,               # 샘플링 확률(probabilistic)
                'top_k': None,              # 샘플링 확률(greedy)
                'temperature': 1.0,          # 샘플링 온도
                'length_penalty': 1,         # 길이 패널티
                'repetition_penalty': 1.0    # 반복 패널티
            }

            # 캡션 생성
            response, _ = model.chat(
                tokenizer=tokenizer,
                msg='',
                user_prompt='Describe the video step by step',
                instruction="Carefully watch the video and describe what is happening in detail.",
                media_type='video',
                media_tensor=frames,
                chat_history=[],
                return_history=True,
                generation_config=generation_config
            )
            # BERTScore을 활용하여 GT와 Prediction 비교. 사용시 아래 2줄 주석 해제 (필요 시, Baseline 평가 Metric으로 활용)
            # P, R, F1 = score([response], [batch['annotations']], lang="en")
            # total_score += F1[0]
            print(f"response: {response}")

    avg_score = total_score / len(dataloader)
    # 만약, BERTScore 등의 평가 메트릭을 사용하여 추가로 total_score를 건드리지 않는다면 avg_score는 계속 0으로 나오는 것이 정상임
    print(F"avg_score: {avg_score}")
    return avg_score

def main():
    # wandb 로그인
    wandb.login()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 모델 경로 설정
    model_path = os.path.join(current_dir, "model/weights")
    
    # 비디오 경로 설정
    video_path = os.path.join(current_dir, "../../data")

    train(model_path, video_path)

if __name__ == "__main__":
    main()
