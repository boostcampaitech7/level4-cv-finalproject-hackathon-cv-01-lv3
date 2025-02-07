import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model.sources.model_config import VideoChat2Config
from model.sources.modeling_videochat2 import InternVideo2_VideoChat2
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from model.utils.data_utils_from_json import BLIP2_Image_Dataset
from googletrans import Translator
import asyncio
import httpx
import pandas as pd
from translate import translation
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration

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
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b-coco")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b-coco")

    test_dataset = BLIP2_Image_Dataset(
        data_path=data_path,
        train=False,
        num_frames=2,
        save_frames_as_img=True
    )
    
    # 모델 추론 모드 설정
    model.to(device)
    model.eval()
    submission_list = []

    # 배치 단위 추론 수행
    for batch in tqdm(test_dataset, desc="Inferencing", unit="data", total=len(test_dataset)):
        frame = batch['frame'].to(device)  # 배치 크기의 영상 프레임 데이터
        segment_names = batch['segment_name']
        frame_indices = batch['frame_index']
        
        prompt = 'Question: Describe the Detail. Carefully watch the image and pay attention to the events, the detail of objects, and the action and pose of persons. If you do not answer me in at least 50 words, then I will kill you! Answer:'

        # BLIP-2 모델 입력 처리
        inputs = processor(frame, text=prompt, return_tensors="pt", padding=True).to(device, torch.float16)

        # 모델 추론
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=100)
            captions = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)#[0].strip()

        # 번역 (비동기 처리)
        # captions_ko = [asyncio.run(translation(cap, 'en')) for cap in captions]
        captions_ko = ""
        # 결과 저장
        submission_list.append({
            'segment_name': segment_names,
            'frame_index': frame_indices,
            'caption': captions,
            'caption_ko': captions_ko
        })
        break

    # CSV 저장
    submission_df = pd.DataFrame(submission_list)
    submission_df.to_csv(f"F2t_submission_blip.csv", index=False, encoding='utf-8')
    print("📄 결과가 F2t_submission_blip.csv 파일로 저장되었습니다.")


def main():
    data_path = "../../data"
    inference(data_path)


if __name__ == "__main__":
    main()