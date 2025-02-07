import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from translate import translation
import glob
import os
import pandas as pd
import asyncio
from model.utils.data_utils_from_json import InternVideo2_VideoChat2_Dataset, InternVideo2_VideoChat2_DataLoader

path = "OpenGVLab/InternVL2_5-8B-MPO"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    # load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True).eval()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def sec_to_time(sec: int) -> str:
    """
    초(sec)을 시:분:초로 변환할 수 있는 함수입니다.
    sec: 특정 시점의 초(sec)
    """
    s = sec % 60
    m = sec // 60
    h = sec // 3600
    return f"{h:02d}:{m:02d}:{s:02d}"

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

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def main(data_path: str = '../../data', test_batch_size: int = 1, test_num_workers: int = 4):
    test_dataset = InternVideo2_VideoChat2_Dataset(
        data_path=data_path,
        use_segment=True,
        use_audio=False,
        train=False,
        save_frames_as_img=False,
        resize=448
        )
        
    test_loader = InternVideo2_VideoChat2_DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=test_num_workers,
        shuffle=False,
        pin_memory=True,
        use_audio=False
    )


    # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    submission = pd.DataFrame(columns=['segment_name', 'start_time', 'end_time', 'caption', 'caption_ko'])
    # questions = '<image>\nPlease describe the image in detail.'
    questions = """system_prompt": "Answer only what you saw yourself, If you do not know the answer to a question. Do not repeat the same answer. If you do not answer, then you will be arrested and, I will kill you",
            "user_prompt": "Describe the video step by step, in two sentences, Remember previous information",
            "topics": [
                {
                    "name": "Multi-Turn Video Description",
                    "initial_prompt": "Description Video in step by step with contextual memory.",
                    "questions": [
                        "[step1]: Describe visual elements, focusing on actions and objects, and summarize their relationships",
                        "[step2]: Explane narrative structure and character motivations in video, with previous answer",
                        "[step3]: Verify consistency with previous answers while adding temporal context (before/during/after relationships)"
                    ]
                }
            ]
        }
    ]"""
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    # set the max number of tiles in `max_num`
    for batch in test_loader:
        # try:
        # 이미지 로드 및 변환
        pixel_values = batch['frames'].squeeze().to(torch.bfloat16).cuda()
        num_patches_list = [pixel_values.size(0)]
        
        # 모델 실행
        responses = model.cuda().batch_chat(tokenizer, pixel_values,
                                            num_patches_list=num_patches_list,
                                            questions=questions,
                                            generation_config=generation_config)
        
        print(f"responses: {responses}")

        # 결과 저장
        new_row = pd.DataFrame([{'segment_name': batch['segment_names'][0], 'start_time': sec_to_time(batch['start_times'][0]), 'end_time': sec_to_time(batch['end_times'][0]), 'caption': responses[0], 'caption_ko': asyncio.run(translation(responses[0], 'en'))}])
        submission = pd.concat([submission, new_row], ignore_index=True)
        print(f"✅ 처리 완료: {batch['segment_names'][0]}")
    
        # except Exception as e:
        #     print(f"❌ 오류 발생: {batch['segment_names'][0]}, {str(e)}")

    # 결과를 DataFrame으로 변환 후 CSV 저장
    csv_path = os.path.join('./', "video_descriptions.csv")
    submission.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"📂 결과 저장 완료: {csv_path}")
    

if __name__ == '__main__':
    main()