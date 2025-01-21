import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model_config import VideoChat2Config
from modeling_videochat2 import InternVideo2_VideoChat2
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from train.utils.data_utils_from_json import InternVideo2_VideoChat2_Dataset, InternVideo2_VideoChat2_DataLoader

def inference(
    json_path,
    model_path,
    video_root,
    test_batch_size=1,
    test_num_workers=4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = VideoChat2Config.from_json_file(
        os.path.join(current_dir, 'config.json')
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_config.llm.pretrained_llm_path,
        trust_remote_code=True,
        use_fast=False,
        token=os.getenv('HF_TOKEN')
    )

    # 모델 초기화
    model = InternVideo2_VideoChat2.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    
    test_dataset = InternVideo2_VideoChat2_Dataset(
        json_path=json_path,
        video_root=video_root,
        use_segment=True,
        use_audio=False,
        train=False
    )
    
    test_loader = InternVideo2_VideoChat2_DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=test_num_workers,
        shuffle=False,
        pin_memory=True,
        use_audio=False
    )

    model.eval()
    for batch in test_loader:
        frames = batch['frames'].to(device)
        outputs = model.chat(
            tokenizer=tokenizer,
            msg='',
            user_prompt="Describe the video step by step",
            instruction="Carefully watch the video and describe what is happening in detail.",
            media_type='video',
            media_tensor=frames,
            chat_history=[],
            return_history=True,
            generation_config={
                'do_sample': False,
                'max_new_tokens': 256,
            }
        )
        print(f"Validation Output: {outputs}")


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

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = "./labels"
    model_path = os.path.join(current_dir)
    # video_path = os.path.join(current_dir.split('tasks/captioning/InternVideo2-Chat-8B/train'), "demo/data")
    video_root = "./data/clips"
    inference(json_path, model_path, video_root)


if __name__ == "__main__":
    main()