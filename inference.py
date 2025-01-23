import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model_config import VideoChat2Config
from modeling_videochat2 import InternVideo2_VideoChat2
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from train.utils.data_utils_from_json import InternVideo2_VideoChat2_Dataset, InternVideo2_VideoChat2_DataLoader
from googletrans import Translator
import asyncio
import httpx
import pandas as pd

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
    json_path: str,
    model_path: str,
    video_root: str,
    test_batch_size: int=1,
    test_num_workers: int=4,
    device: str='cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
    """
    InternVideo2 모델을 활용하여 Inference하는 함수입니다.
    ---------------------------------------------------
    args
    json_path: JSON형태의 레이블이 있는 경로
    model_path: InternVideo2 모델이 있는 경로
    video_root: mp4형태의 비디오가 있는 경로
    test_batch_size: Batch Size
    test_num_workers: num_workers 수 설정
    device: cpu 혹은 cuda 등 Inference를 수행할 주체를 설정

    출력: 'segment_name', 'start_time', 'end_time', 'caption', 'caption_ko'로 이루어져 있는 v2t_submission.csv
    """
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
    submission = pd.DataFrame(columns=['segment_name', 'start_time', 'end_time', 'caption', 'caption_ko'])
    for batch in test_loader:
        frames = batch['frames'].to(device)
        outputs = model.chat(
            tokenizer=tokenizer,
            msg='',
            user_prompt="Describe the video in one sentence",
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
        new_row = pd.DataFrame([{'segment_name': batch['segment_names'][0], 'start_time': sec_to_time(batch['start_times'][0]), 'end_time': sec_to_time(batch['end_times'][0]), 'caption': outputs[0].strip(), 'caption_ko': asyncio.run(translation(outputs[0], 'en'))}])
        submission = pd.concat([submission, new_row], ignore_index=True)
    submission.to_csv(f"v2t_submission.csv", index=False)
    

async def translation(caption: str, typ: str) -> str:
    """
    번역을 수행하는 함수
    주의 사항: 이 함수는 asynchronous하게 작동합니다
    --------------------
    args
    caption: 번역할 문장 또는 단어
    typ: 번역할 문장의 언어 (한국어로 입력 받을 시 ko, 영어로 입력 받을 시 en으로 설정)

    출력: 번역된 문장 또는 단어 (str)
    """
    translator = Translator()
    if typ == 'ko':
        src, dest = 'ko', 'en'
    if typ == 'en':
        src, dest = 'en', 'ko'
    retries = 5  
    for attempt in range(retries):
        try:
            res = await translator.translate(caption, src=src, dest=dest)
            return res.text
        except httpx.ConnectTimeout:
            print(f"Connection timeout occurred. Retry {attempt + 1} of {retries}...")
            await asyncio.sleep(2)  # 대기 후 재시도
    raise RuntimeError(f'Translation Failed for {retries} times')

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "data/labels")
    model_path = os.path.join(current_dir)
    video_root = os.path.join(current_dir, "data/clips")
    inference(json_path, model_path, video_root)


if __name__ == "__main__":
    main()