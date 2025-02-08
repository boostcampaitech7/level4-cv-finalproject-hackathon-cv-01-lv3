import torch
from transformers import AutoTokenizer, AutoModel
from translate import translation
import os
import pandas as pd
import asyncio
from model.utils.data_utils_from_json import InternVL_Video_Dataset, InternVL_Video_DataLoader
from tqdm import tqdm
import time
import json

path = "OpenGVLab/InternVL2_5-8B-MPO"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    # load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True).eval()

def sec_to_time(sec: int) -> str:
    """
    초(sec)을 시:분:초로 변환할 수 있는 함수입니다.
    sec: 특정 시점의 초(sec)
    """
    s = sec % 60
    m = sec // 60
    h = sec // 3600
    return f"{h:02d}:{m:02d}:{s:02d}"

def time_to_seconds(time_str):
    """HH:MM:SS 형식을 초 단위로 변환"""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def load_summary_json(summary_dir: str, video_name: str):

    data_path = os.path.join(summary_dir, f"{video_name}.json")
    with open(data_path, 'r') as f:
        data = json.load(f)
    model_output = json.loads(data["model_output"])
    return model_output["Genre"] , model_output["Summary"]


def get_speech_caption(vision_caption_path, speech_caption_path):
    # 비전 캡션 로드
    with open(vision_caption_path, 'r') as f:
        vision_data = json.load(f)
    
    # 음성 캡션 로드
    with open(speech_caption_path, 'r') as f:
        speech_data = json.load(f)

    # 비전 타임라인 추출
    video_id = next(iter(vision_data))  # 첫 번째 키 추출 (e.g. "yt8m_Movieclips_xcJXT5lc1Bg_001")
    vision_start = time_to_seconds(vision_data[video_id]['start_time'])
    vision_end = time_to_seconds(vision_data[video_id]['end_time'])

    # 시간대 필터링
    overlapping_speech = []
    for speech in speech_data:
        speech_start = time_to_seconds(speech['start_time'])
        speech_end = time_to_seconds(speech['end_time'])
        
        # 시간대 겹침 조건 (부분 겹침 포함)
        if (speech_start < vision_end) and (speech_end > vision_start):
            overlapping_speech.append(speech['speech_cap'])

    # 캡션 통합
    merged_caption = ' '.join(overlapping_speech)
    
    return merged_caption


def main(data_path: str = '../../data', test_batch_size: int = 1, test_num_workers: int = 4):
    test_dataset = InternVL_Video_Dataset(
        data_path=data_path,
        train=False,
        save_frames_as_img=False,
        input_size=448,
        num_frames=16
        )
        
    test_loader = InternVL_Video_DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=test_num_workers,
        shuffle=False,
        pin_memory=True,
    )


    # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    submission = pd.DataFrame(columns=['segment_name', 'start_time', 'end_time', 'caption', 'caption_ko'])
    # questions = '<image>\nPlease describe the image in detail.'
    summary_dir = os.path.join(data_path, 'YT8M', 'Movieclips', 'test', 'summary_json')
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    # set the max number of tiles in `max_num`
    before_time = time.time()
    for batch in tqdm(test_loader, desc="Processing", total=len(test_loader), unit="batch"):
        # 이미지 로드 및 변환
        pixel_values, num_patches_list = batch['pixel_values'], batch['num_patches_lists'][0]
        pixel_values = pixel_values.squeeze().to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        
        # 요약
        seg_name = batch['segment_names'][0]# edit(0208)
        video_name = "_".join(seg_name.split("_")[:-1])
        genre , video_summary = load_summary_json(summary_dir, video_name)
        vision_json_path = os.path.join(data_path, 'YT8M', 'Movieclips', 'test', 'labels', f'{batch["segment_names"][0]}.json')
        base_name = '_'.join(batch["segment_names"][0].split('_')[:-1])
        # 음성
        speech_json_path = os.path.join(data_path, 'YT8M', 'Movieclips', 'test', 'stt', f'{base_name}.json')
        speech = get_speech_caption(vision_json_path, speech_json_path)
        print(batch["segment_names"][0], speech)
        fusion_prompt = f"""<instruction> Answer only what you observed in the video clip. Do not repeat the same answer. Describe the video step by step. 
            Do not avoid answering, Answer only what you saw yourself, If you do not know the answer to a question.
            <information> {video_summary}, Genre of the video: {genre}, use overall context only to better understand the story </information>
            <speech> {speech} 
            <question> Describe the action and object(human, items, natural, etc) in this video. Include some desciption of sppech information yourself.  
            <request> Only answer in one sentences, but it also includes essential information from the video. 
        """
        prompt = video_prefix + fusion_prompt

        responses = model.cuda().chat(tokenizer, pixel_values,
                                            num_patches_list=num_patches_list,
                                            question=prompt,
                                            generation_config=generation_config)
        # 결과 저장
        print(responses)
        new_row = pd.DataFrame([{'segment_name': batch['segment_names'][0], 'start_time': batch['start_times'][0], 'end_time': batch['end_times'][0], 'caption': responses, 'caption_ko': asyncio.run(translation(responses, 'en'))}])
        submission = pd.concat([submission, new_row], ignore_index=True)
    
    # 결과를 DataFrame으로 변환 후 CSV 저장
    after_time = time.time()
    csv_path = os.path.join('./', "v2t_submissions_InternVL2-5.csv")
    submission.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"📂 결과 저장 완료: {csv_path}")
    print(f"⏰ Inference 소요 시간: {sec_to_time(int(after_time - before_time))}")
    

if __name__ == '__main__':
    main()