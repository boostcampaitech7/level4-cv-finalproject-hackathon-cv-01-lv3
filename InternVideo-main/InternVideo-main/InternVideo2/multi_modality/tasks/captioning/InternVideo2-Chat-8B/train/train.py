import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model_config import VideoChat2Config
from modeling_videochat2 import InternVideo2_VideoChat2
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from utils.data_utils import InternVideo2_VideoChat2_Dataset, InternVideo2_VideoChat2_DataLoader

def train(
    model_path,
    video_path,
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
        os.path.join(current_dir.split('/utils'), 'config.json')
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_config.llm.pretrained_llm_path,
        trust_remote_code=True,
        use_fast=False,
        token=os.getenv('HF_TOKEN')
    )

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
        video_path=video_path,
        use_segment=True,
        use_audio=False,
        train=True
    )
    
    test_dataset = InternVideo2_VideoChat2_Dataset(
        video_path=video_path,
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
        use_audio=False,
        train=True
    )
    
    test_loader = InternVideo2_VideoChat2_DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=test_num_workers,
        shuffle=False,
        pin_memory=True,
        use_audio=False,
        train=False 
    )
    
    optimizer, scheduler = model.prepare_for_training(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # ROUGE Score를 사용한 텍스트 유사도 측정
    criterion = torch.nn.CrossEntropyLoss() 
    from rouge_score import rouge_scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            if torch.cuda.is_available():
                batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model.chat(
                tokenizer=tokenizer,
                msg='',
                user_prompt="Describe the video step by step",
                instruction="Carefully watch the video and describe what is happening in detail.",
                media_type='video',
                chat_history=[],
                return_history=True,
                generation_config={
                    'do_sample': False,
                    'max_new_tokens': 256,
                }
            )
            loss = criterion(outputs, batch['annotation'])
            loss.backward()
            optimizer.step()
            scheduler.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
            
            if epoch % validation_interval == 0:
                validation(model, test_dataset, test_loader, video_path, tokenizer, device)
                

def validation(model, dataset, dataloader, video_path, tokenizer, device, criterion, rouge):
    model.eval()
    
    print(f"Validation start")
    for batch in dataloader:
        if torch.cuda.is_available():
            batch = batch.to(device)
        
        outputs = model.chat(
            tokenizer=tokenizer,
            msg='',
            user_prompt="Describe the video step by step",
            instruction="Carefully watch the video and describe what is happening in detail.",
            media_type='video',
            chat_history=[],
            return_history=True,
            generation_config={
                'do_sample': False,
                'max_new_tokens': 256,
            }
        )
        
        print(f"Validation Output: {outputs}")
        print(f"Annotation: {batch['annotation']}")
        print(f"Loss: {criterion(outputs, batch['annotation'])}")
        print(f"ROUGE Score: {rouge.score(outputs, batch['annotation'])}")
        print(f"Validation end")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir.split('train'))
    video_path = os.path.join(current_dir.split('tasks/captioning/InternVideo2-Chat-8B/train'), "demo/data")
    train(model_path, video_path)
    