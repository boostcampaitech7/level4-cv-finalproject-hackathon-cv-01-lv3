import torch
from transformers import AutoTokenizer, AutoModel
import os
import pandas as pd
import asyncio
from tqdm import tqdm
import time
import json
import sys
# project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from model.utils.data_utils_from_json import InternVL_Video_Dataset, InternVL_Video_DataLoader
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from translate import translation


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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
    model_output = json.loads(data)
    return model_output["Genre"] , model_output["Summary"]


def get_speech_caption(vision_caption_path, speech_caption_path):
    # 비전 캡션 로드
    with open(vision_caption_path, 'r') as f:
        vision_data = json.load(f)
    
    # 음성 캡션 로드
    with open(speech_caption_path, 'r') as f:
        speech_data = json.load(f)
    speech_data = speech_data['result']
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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

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
    generation_config = dict(max_new_tokens=512, do_sample=False)
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
        new_row = pd.DataFrame([{'segment_name': batch['segment_names'][0], 'start_time': batch['start_times'][0], 'end_time': batch['end_times'][0], 'caption': responses, 'caption_ko': asyncio.run(translation(responses, 'en'))}])
        submission = pd.concat([submission, new_row], ignore_index=True)

    # 결과를 DataFrame으로 변환 후 CSV 저장
    after_time = time.time()
    csv_path = os.path.join('./', "v2t_submissions_InternVL2-5.csv")
    submission.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"📂 결과 저장 완료: {csv_path}")
    print(f"⏰ Inference 소요 시간: {sec_to_time(int(after_time - before_time))}")
    

class InternVL():
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            "OpenGVLab/InternVL2_5-8B-MPO",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True
        ).eval().cuda()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "OpenGVLab/InternVL2_5-8B-MPO",
            trust_remote_code=True,
            use_fast=False
        )
        self.generation_config = dict(max_new_tokens=1024, do_sample=False)

    def load_media(self, media_path: str, media_type: str = 'video', **kwargs):
        """기존 DataLoader 로직을 활용한 미디어 로드"""
        # 기존 데이터셋/데이터로더 초기화 방식 유지
        dataset = InternVL_Video_Dataset(
            data_path=os.path.dirname(os.path.dirname(media_path)),  
            train=False,
            save_frames_as_img=False,
            input_size=448,
            num_frames=16
        )
        
        loader = InternVL_Video_DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )
        
        for batch in loader:
            return batch['pixel_values'].squeeze(), batch['num_patches_lists'][0]
        return None, None

    def generate_caption_image(self, media_tensor, media_type: str):
        """기존 프롬프트 생성 로직을 활용한 캡션 생성"""
        pixel_values= media_tensor
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        num_patches_list = [pixel_values.size(0)]


        fusion_prompt = f""" 
                        <instruction> Answer only what you observed in the video clip.
                        Do not repeat the same answer. Describe the video step by step.
                        Do not avoid answering, Answer only what you saw yourself, If you do not know the answer to a question.
                        <question> Describe the action and object(human, items, natural, etc) in this video.    
                        <request> Only answer in one sentences, but it also includes essential information from the video
                            """

        # 생성 파라미터 조정
        generation_config = dict(
            max_new_tokens=1024,  # 토큰 길이 증가
            do_sample=False,      
            temperature=1      # 창의성 조절
        )

        responses = self.model.chat(
            self.tokenizer,
            pixel_values,
            num_patches_list=num_patches_list,
            question=fusion_prompt,
            generation_config=generation_config 
        )
        print(f"\n=== 생성된 캡션 ===")
        print(responses)
        return responses

    def generate_caption(self, media_tensor, media_type: str, vision_caption_json_path: str, speech_caption_json_path: str, summary_dir: str, video_name: str):
        """기존 프롬프트 생성 로직을 활용한 캡션 생성"""
        pixel_values, num_patches_list = media_tensor
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        
        # 프롬프트 구성 전에 summary 로드 확인
        try:
            genre, video_summary = load_summary_json(summary_dir, video_name)  # 순서 수정
        except Exception as e:
            video_summary = "No summary available"
            genre = "Unknown"

        # 음성 캡션 로드 확인
        try:
            speech = get_speech_caption(vision_caption_json_path, speech_caption_json_path)
            if not speech:
                speech = "No speech detected"
        except Exception as e:
            print(f"음성 캡션 로드 실패: {str(e)}")
            speech = "No speech available"


        # 기존 프롬프트 구성 방식 유지
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

        fusion_prompt = f"""<instruction> Answer only what you observed in the video clip. Do not repeat the same answer. Describe the video step by step. 

            Do not avoid answering, Answer only what you saw yourself, If you do not know the answer to a question.
            <information> {video_summary}, Genre of the video: {genre}, use overall context only to better understand the story </information>
            <speech> {speech} 
            <question> Describe the action and object(human, items, natural, etc) in this video. Include some desciption of speech information yourself.  
            <request> Only answer in one sentences, but it also includes essential information from the video. 
        """

        # 생성 파라미터 조정
        generation_config = dict(
            max_new_tokens=1024,  # 토큰 길이 증가
            do_sample=False,      
            temperature=1      # 창의성 조절
        )

        responses = self.model.chat(
            self.tokenizer,
            pixel_values,
            num_patches_list=num_patches_list,
            question=video_prefix + fusion_prompt,
            generation_config=generation_config 
        )
        print(f"\n=== 생성된 캡션 ===")
        print(responses)
        return responses
        
    @staticmethod  # 정적 메서드로 변경
    def load_single_video(video_path, bound=None, input_size=448, max_num=1, num_segments=16):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

if __name__ == '__main__':
    main()