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
import transformers
from datetime import datetime

def train(
    model_path,
    data_path="../../data",
    num_epochs=100,
    train_batch_size=2,
    test_batch_size=1,
    train_num_workers=4,
    test_num_workers=4,
    learning_rate=5e-5,
    weight_decay=1e-2,
    validation_interval=1,
    device='cuda',
    log_file='train_log.txt'
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = VideoChat2Config.from_json_file(
        os.path.join(current_dir,'model','configs', 'config.json')
    )

    # 토크나이저 초기화 (Mistral-7B) (LlamaTokenizer 사용)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_config.llm.pretrained_llm_path,
        trust_remote_code=True,
        use_fast=False,
        token=os.getenv('HF_TOKEN')
    )
    if '[PAD]' not in tokenizer.get_vocab():
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

    # 모델 내부적으로 Attention, Loss, Output 출력 등에서 pad_token이 불필요한 영향을 주지 않도록 설정 (이 코드를 넣든 말든 둘 다 attention mask는 적용되던데 일단 추가함)
    model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = InternVideo2_VideoChat2_Dataset(
        data_path=data_path,
        use_segment=True,
        use_audio=False,
        train=True,
        num_frames=12
    )

    test_dataset = InternVideo2_VideoChat2_Dataset(
        data_path=data_path,
        use_segment=True,
        use_audio=False,
        train=False,
        num_frames=12
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

    from rouge_score import rouge_scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    query_embedding_size = model.query_tokens.shape[1] + model.extra_query_tokens.shape[1]
    # video_index = torch.ones(train_batch_size, query_embedding_size).to(device)

    # 학습 도중에 Text와 Video의 각 embedding vector를 합치는 과정이 있습니다.
    # 현재 알고리즘은, video_index 크기를 text embedding vector보다 작게 유지시켜야 합니다.
    # (내부 알고리즘에서 video_index를 text에 맞춤)
    # 따라서 추후 논의 전까지, raise 검증문을 하나 추가합니다.
    if train_batch_size * query_embedding_size > 300:
        raise ValueError("upper bound of Video index is 300, video_index=train_batch_size*query_embedding_size")

    best_loss = float('inf')
    for epoch in range(num_epochs):
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(train_loop):
            if torch.cuda.is_available():
                frames = batch['frames'].to(device)
                annotations = batch['annotations']  # 텍스트 데이터는 그대로 유지
            else:
                frames = batch['frames']
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
            # forward 패스 수행
            # 현재는 LLM 출력이 Outputs에 해당하고, LoRA를 건들면 제대로 값이 나오질 않으니, 이대로 갑니다. 
            # 추후 가중치 변경을 잘 끝내면 Q-former에 대한 logit 계산을 위해 validation처럼 text_embeds를 사용하겠습니다. 
            # forward 함수를 호출하면서, attention_mask는 text에 대한 logit을 만들 때 사용함
            outputs, _ = model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                video=frames,
                labels=text_inputs.input_ids,
                video_idx=torch.ones(frames.shape[0], query_embedding_size).to(device)
            )
            # print(f"type(outputs): {type(outputs)}, outputs.loss: {type(outputs.loss)}, {outputs.loss}")
            loss = outputs.loss
            if log_file is not None:
                with open(log_file, "a") as f:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"EPOCH {epoch+1}/{num_epochs} BATCH {batch_idx+1}/{len(train_loader)} [{current_time}] Loss: {loss:.4f}\n")
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print(f"outputs: {outputs[0]}")
            # tqdm description 업데이트하여 loss 표시
            train_loop.set_postfix(loss=f'{loss.item():.4f}, batch_idx: {batch_idx}')
            
            del frames
            del annotations
            del text_inputs
            del outputs
            del loss

            # Cache는 내부에서 사용해서, 지울 수 없음.

        if epoch % validation_interval == 0:
            print("--------------------------------")
            print(f"validation start, epoch: {epoch+1}")
            avg_loss = validation(model, test_loader, tokenizer, device, query_embedding_size, log_file=log_file)
            if best_loss > avg_loss:
                print(f"Best loss changed from {best_loss:.4f} -> {avg_loss:.4f}")
                save_model(model, optimizer=optimizer, epoch=epoch, loss=avg_loss, save_path=os.path.join('temp_model', 'best_model.pt'))
                best_loss = avg_loss
            print(f"validation end, epoch: {epoch+1}")
            print("--------------------------------")
            model.train()
            
def save_model(model, optimizer=None, scheduler=None, epoch=None, loss=None, save_path="best_model.pt"):
    """
    Save the model's state_dict along with optional training metadata.

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



def validation(model, dataloader, tokenizer, device, query_embedding_size, log_file=None):
    model.eval()
    total_loss = 0
    # validation loop에도 tqdm 적용
    val_loop = tqdm(dataloader, desc='Validation')
    with torch.no_grad():
        for batch in val_loop:
            frames = batch['frames'].to(device)
            annotations = batch['annotations']
            
            text_inputs = tokenizer(
                annotations,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)
            
            outputs, _ = model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                video=frames,
                labels=text_inputs.input_ids,
                video_idx=torch.ones(frames.shape[0], query_embedding_size).to(device)
            )
            # print(f"outputs type: {type(outputs)}, length: {len(outputs)}")
            # decoded_texts = tokenizer.decode(outputs, skip_special_tokens=True)
            # print(f"Decoded text: {decoded_texts}")

            ### 임시
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            predicted_ids = logits.argmax(dim=-1)  # 가장 높은 확률의 토큰 ID 선택
            total_loss += outputs.loss
            # 토크나이저로 디코딩
            for predicted_id in predicted_ids:
                decoded_text = tokenizer.decode(predicted_id, skip_special_tokens=True)
                print(f"Decoded text: {decoded_text}")
            ###

            generation_config = {
                'num_beams': 1,            # 빔 서치 크기
                'max_new_tokens': 200,     # 최대 생성 길이
                'do_sample': False,         # 샘플링 사용
                'top_p': 0.9,
                'top_k': None,
                'temperature': 1.0,
                'length_penalty': 1,
                'repetition_penalty': 1.0
            }

            # generate를 사용하기에는 decoding에 관한 이해가 전혀 없어 pass. 다시 chat 기반으로 수정함.
            # 캡션 생성
            # response, _ = model.chat(
            #     tokenizer=tokenizer,
            #     msg='',
            #     user_prompt='Describe the video step by step',
            #     instruction="Carefully watch the video and describe what is happening in detail.",
            #     media_type='video',
            #     media_tensor=frames,
            #     chat_history=[],
            #     return_history=True,
            #     generation_config=generation_config
            # )
            
            # print(f"Validation Output: {response}")
            # print(f"Annotation: {batch['annotations']}")

    avg_loss = total_loss / len(dataloader)
    # print(f"Validation Loss: {avg_loss}")
    if log_file is not None:
        with open(log_file, "a") as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{current_time}] Validation Loss: {avg_loss:.4f}\n")
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
