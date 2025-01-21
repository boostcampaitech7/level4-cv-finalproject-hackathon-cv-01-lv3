import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model_config import VideoChat2Config
from modeling_videochat2 import InternVideo2_VideoChat2
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from train.utils.data_utils import InternVideo2_VideoChat2_Dataset, InternVideo2_VideoChat2_DataLoader

def train(
    model_path,
    video_path,
    csv_path,
    num_epochs=10,
    train_batch_size=2,
    test_batch_size=1,
    train_num_workers=4,
    test_num_workers=4,
    learning_rate=1e-4,
    weight_decay=1e-2,
    validation_interval=1,
    device='cuda'
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = VideoChat2Config.from_json_file(
        os.path.join(current_dir, 'config.json')
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
        csv_path=csv_path,
        video_root=video_path,
        use_segment=True,
        use_audio=False,
        train=True
    )
    
    test_dataset = InternVideo2_VideoChat2_Dataset(
        csv_path=csv_path,
        video_root=video_path,
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
    # # text_inputs.input_ids의 shape를 기반으로 video_idx 생성
    # batch_size, seq_length = text_inputs.input_ids.shape
    # video_idx = torch.zeros((batch_size, seq_length)).to(device)

    # # 비디오 토큰을 위한 96개의 위치만 1로 설정
    # # 예: 처음 96개 위치에 비디오 토큰 배치
    # video_idx[:, :96] = 1

    from rouge_score import rouge_scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            if torch.cuda.is_available():
                frames = batch['frames'].to(device)
                annotations = batch['annotations']  # 텍스트 데이터는 그대로 유지
            else:
                frames = batch['frames']
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

            # forward 패스 수행
            outputs = model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                video=frames,
                labels=text_inputs.input_ids,
                # video_idx=torch.ones(2, 96).to(device),
                video_idx=torch.ones_like(text_inputs.input_ids)  # 비디오 토큰 위치 표시
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()


            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
            
            if epoch % validation_interval == 0:
                validation(model, test_dataset, test_loader, video_path, tokenizer, device)
                model.train()
                

def validation(model, dataset, dataloader, tokenizer, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            frames = batch['frames'].to(device)
            annotations = batch['annotations']
            
            text_inputs = tokenizer(
                annotations,
                padding='longest',
                truncation=True,
                max_length=192,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                video=frames,
                labels=text_inputs.input_ids,
                # video_idx=torch.ones(2, 96).to(device),
                # video_idx=torch.tensor([1] * text_inputs.input_ids.shape[0]).to(device)
                video_idx=torch.ones_like(text_inputs.input_ids)
            )
            
            total_loss += outputs.loss.item()
    
    print(f"Validation Output: {outputs}")
    print(f"Annotation: {batch['annotations']}")

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss}")
    return avg_loss

def main():
    # 상위 디렉토리로 이동하여 필요한 경로 생성
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))  # multi_modality 폴더까지
    # 모델 경로 설정
    model_path = "/data/ephemeral/home/deamin/project/level4-cv-finalproject-hackathon-cv-01-lv3/InternVideo-main/InternVideo-main/InternVideo2/multi_modality/tasks/captioning/InternVideo2-Chat-8B"
    
    # 비디오 경로 설정
    video_path = "/data/ephemeral/home/deamin/project/level4-cv-finalproject-hackathon-cv-01-lv3/InternVideo-main/InternVideo-main/InternVideo2/multi_modality/demo/data"
    
    csv_path = "/data/ephemeral/home/deamin/project/level4-cv-finalproject-hackathon-cv-01-lv3/InternVideo-main/InternVideo-main/InternVideo2/multi_modality/demo/data/internVideo2_dataformat_011725.csv"

    train(model_path, video_path, csv_path)
    

if __name__ == "__main__":
    main()
    