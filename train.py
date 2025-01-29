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

def train(
    model_path: str,
    video_path: str,
    data_path: str = "../../data",
    num_epochs: int = 50,
    train_batch_size: int = 2,
    test_batch_size: int = 1,
    train_num_workers: int = 4,
    test_num_workers: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    validation_interval: int = 25,
    device: str = None
):
    """
    모델을 학습시키는 함수, 일정 주기로 validation을 수행함.

    Args: 
        model_path: 모델 경로
        video_path: 비디오 경로
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
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    if torch.cuda.is_available():
        model = InternVideo2_VideoChat2.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)
    else:
        model = InternVideo2_VideoChat2.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

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
        shuffle=False,
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

    for epoch in range(num_epochs):
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(train_loop):

            frames = batch['frames'].to(device)
            annotations = batch['annotations']
            
            # 텍스트 토큰화
            text_inputs = tokenizer(
                annotations,
                padding='longest',
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
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loop.set_postfix(loss=f'{loss.item():.4f}, batch_idx: {batch_idx}')
            
            del frames, annotations, text_inputs, outputs, loss

        if epoch % validation_interval == 0:
            print("--------------------------------")
            print(f"validation start, epoch: {epoch+1}")
            validation(model, test_loader, tokenizer, device, query_embedding_size)
            print(f"validation end, epoch: {epoch+1}")
            print("--------------------------------")
            model.train()
            

def validation(model, dataloader, tokenizer, device, query_embedding_size):
    model.eval()
    total_loss = 0
    val_loop = tqdm(dataloader, desc='Validation')
    with torch.no_grad():
        for batch in val_loop:
            frames = batch['frames'].to(device)
            annotations = batch['annotations']
            
            text_inputs = tokenizer(
                annotations,
                padding='longest',
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)
            
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
                'top_p': 0.9,               # 샘플링 확률
                'top_k': None,              # 샘플링 확률
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
            
            print(f"Validation Output: {response}")
            print(f"Annotation: {batch['annotations']}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    # 현재 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 모델 경로 설정
    model_path = os.path.join(current_dir, "model/weights")
    
    # 비디오 경로 설정
    video_path = os.path.join(current_dir, "../../data")

    train(model_path, video_path)

if __name__ == "__main__":
    main()
