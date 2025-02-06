import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model.sources.model_config import VideoChat2Config
from model.sources.modeling_videochat2 import InternVideo2_VideoChat2
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from model.utils.data_utils_from_json import InternVideo2_VideoChat2_Dataset, InternVideo2_VideoChat2_DataLoader, InternVideo2_VideoChat2_Image_Dataset, InternVideo2_VideoChat2_Image_DataLoader
from googletrans import Translator
import asyncio
import httpx
import pandas as pd
from translate import translation
from tqdm import tqdm

def sec_to_time(sec: int) -> str:
    """
    초(sec)을 시:분:초로 변환할 수 있는 함수입니다.
    sec: 특정 시점의 초(sec)
    """
    s = sec % 60
    m = sec // 60
    h = sec // 3600
    return f"{h:02d}:{m:02d}:{s:02d}"

def inference(
    data_path: str,
    model_path: str,
    test_batch_size: int=1,
    test_num_workers: int=4,
    device: str='cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
    """
    InternVideo2 모델을 활용하여 Inference하는 함수입니다.
    ---------------------------------------------------
    args
    data_path: 데이터가 있는 경로
    model_path: InternVideo2 모델이 있는 경로
    test_batch_size: Batch Size
    test_num_workers: num_workers 수 설정
    device: cpu 혹은 cuda 등 Inference를 수행할 주체를 설정

    출력: 'segment_name', 'start_time', 'end_time', 'caption', 'caption_ko'로 이루어져 있는 v2t_submission.csv
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = VideoChat2Config.from_json_file(
        os.path.join(current_dir, 'model', 'configs', 'config.json')
    )

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
    model.config.pad_token_id = tokenizer.pad_token_id
    test_dataset = InternVideo2_VideoChat2_Image_Dataset(
        data_path=data_path,
        use_segment=True,
        use_audio=False,
        train=False,
        num_sampling=2,
        save_frames_as_img=True
    )
    
    test_loader = InternVideo2_VideoChat2_Image_DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=test_num_workers,
        shuffle=False,
        pin_memory=True,
    )
    model.eval()
    submission = pd.DataFrame(columns=['segment_name', 'start_time', 'end_time', 'caption', 'caption_ko'])
    for batch in tqdm(test_loader, desc="Inferencing", unit="batch", total=len(test_loader)):
        frames = batch['frames'].to(device)
        outputs = model.chat(
            tokenizer=tokenizer,
            msg='',
            user_prompt='Describe the image in detail.',
            instruction="Carefully watch the image and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons.",
            media_type='image',
            media_tensor=frames,
            chat_history=[],
            return_history=True,
            generation_config={
                'do_sample': False,
                'max_new_tokens': 256,
            }
        )
        new_row = pd.DataFrame([{'segment_name': batch['segment_names'][0], 'frame_index': batch['frame_indices'][0], 'caption': outputs[0].strip(), 'caption_ko': asyncio.run(translation(outputs[0], 'en'))}])
        submission = pd.concat([submission, new_row], ignore_index=True)
    submission.to_csv(f"F2t_submission_new.csv", index=False, encoding='utf-8')
    


def main():
    data_path = "../../data"
    model_path = "./model/weights"
    inference(data_path, model_path)


if __name__ == "__main__":
    main()