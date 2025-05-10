from model_internvl3.InternVL3 import load_video
import torch
from transformers import AutoModel, AutoTokenizer
import os

# If you set `load_in_8bit=True`, you will need two 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
path = '../../t2v/weights/OpenGVLab/InternVL3-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

def process_segment(video_path, max_new_tokens=1024, do_sample=True, num_segments=16, max_num=1):
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    video_path = "../YT8M/clips/yt8m_Movieclips__8LrZ4NhPmk_001.mp4" if video_path is None else video_path
    pixel_values, num_patches_list = load_video(video_path, num_segments=16, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    print(f"pixel_values.shape: {pixel_values.shape}") # 추후 삭제 예정

    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + 'Describe the Video in very Detail.'

    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list, history=None, return_history=True)
    return response

clip_path = "D:/YT8M_backup/clips"
full_video_id = "nSJxx_KUEes"
full_video = sorted([os.path.join(clip_path, x) for x in os.listdir(clip_path) if x.startswith(f"yt8m_Movieclips_{full_video_id}")])

print(full_video)