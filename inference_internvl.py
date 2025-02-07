import torch
from transformers import AutoTokenizer, AutoModel
from translate import translation
import os
import pandas as pd
import asyncio
from model.utils.data_utils_from_json import InternVL_Video_Dataset, InternVL_Video_DataLoader
from tqdm import tqdm
import time


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
    
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    # set the max number of tiles in `max_num`
    before_time = time.time()
    for batch in tqdm(test_loader, desc="Processing", total=len(test_loader), unit="batch"):
        # 이미지 로드 및 변환
        pixel_values, num_patches_list = batch['pixel_values'], batch['num_patches_lists'][0]
        pixel_values = pixel_values.squeeze().to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        
        genre = "Movie"
        video_summary =  "The video shows scenes from a movie, including a car driving at night, people in a car, and a store scene. It also includes credits and a website link."
        speech = '''<merged_speech>: "So how much do I get paid? 25 bucks a car? Paid? You don't get paid. You kidding? You work on commission. That's better than being paid. Most cars you rip are worth two or three hundred dollars. A $50,000 Porsche might make you five grand. Come on, dickhead."
        '''
        
        fused_prompt = f""" {{
        "system_prompt": "Answer only what you observed in the video clip. Provide a detailed and complete description in at least two to three sentences. Do not repeat the same answer or evade the question.",
        "user_prompt": "Describe the video clip step by step in two to three sentences. Clearly detail the visual elements, actions, objects, and their relationships. Overall video context for reference 
        (note: the video summary contains details about the full video and may include events or locations not present in this clip): Genre: {genre}, Video Summary: {video_summary}.
        Remember to focus on the clip content, and use the overall context only to better understand the story.",
        
        "topics": [
            {{
            "name": "Multi-Turn Video Description",
            "initial_prompt": "Provide a detailed, step-by-step description of the video clip, incorporating all relevant details observed in the clip. Use the overall video context as reference only if necessary.",
            "questions": [
                "[step1]: Describe the visual elements, focusing on actions, objects, and their relationships.",
                "[step2]: Explain the narrative structure and character motivations based solely on the clip.",
                "[step3]: Verify consistency with earlier responses by adding temporal context (before, during, and after events observed in the clip).",
                "[step4]: Summarize the previous steps into a cohesive paragraph that maintains logical continuity. Remember, the overall video context is provided only as a reference for understanding the full story, and may not reflect all details present in this clip."
            ]
            }}
        ]
        }}"""
        
        prompt = video_prefix + fused_prompt

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
    

if __name__ == '__main__':
    main()