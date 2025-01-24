import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model.sources.model_config import VideoChat2Config
from model.sources.modeling_videochat2 import InternVideo2_VideoChat2
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from model.utils.data_utils_from_json import InternVideo2_VideoChat2_Dataset, InternVideo2_VideoChat2_DataLoader
from googletrans import Translator
import asyncio
import httpx
import pandas as pd
# clipping video
from data.utils.clip_video import split_video_into_scenes
from inference import inference

def sec_to_time(sec: int) -> str:
    """
    초(sec)을 시:분:초로 변환할 수 있는 함수입니다.
    sec: 특정 시점의 초(sec)
    """
    s = sec % 60
    m = sec // 60
    h = sec // 3600
    return f"{h:02d}:{m:02d}:{s:02d}"


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
    model_path = os.path.join(current_dir, "model/weights")
    
    # !!!경로 설정필요!!!
    # video_dir: 원본 비디오 경로 (디렉토리)
    # json_dir: 라벨 저장할 경로 (디렉토리)
    # seg_dir: 장면 저장할 경로 (디렉토리)

    video_dir = os.path.join(current_dir,"../../data/YT8M/movieclips/origin")
    json_dir = os.path.join(current_dir,"../../data/YT8M/movieclips/labels")
    seg_dir = os.path.join(current_dir,"../../data/YT8M/movieclips/clips")

    # 1. video clipping 
    for file_name in os.listdir(video_dir):
        if file_name.lower().endswith('.mp4'):
            video_path = os.path.join(video_dir, file_name)
            split_video_into_scenes(
                video_path=video_path,
                threshold=27.0,
                output_json_dir=json_dir,
                segments_dir=seg_dir
            )    

    
    # 2. seg captioning & translate
    inference(json_dir, model_path, seg_dir)#video_root


if __name__ == "__main__":
    main()
