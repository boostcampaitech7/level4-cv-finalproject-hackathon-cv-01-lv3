import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
import cv2
import pandas as pd
import os
from tqdm import tqdm
import time
import json
import subprocess

def get_adaptive_fps(video_path, target_frames=16):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    if fps is None or fps <= 0 or frame_count <= 0:
        return 1  # fallback
    duration = frame_count / fps
    return target_frames / duration


def main():
    model_name = "./weights/DAMO-NLP-SG/VideoLLaMA3-7B"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    df = pd.read_csv("./sample_evaluation.csv")

    video_path = "./YT8M/clips/yt8m_Movieclips__8LrZ4NhPmk_006.mp4"
    question = "Describe the video in very detail."

    captions = []

    for p_video_path in tqdm(df['Segment']):
        video_path = os.path.join("./YT8M/clips", f"{p_video_path}.mp4")
        # Video conversation
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": get_adaptive_fps(video_path), "max_frames": 16}},
                    {"type": "text", "text": question},
                ]
            },
        ]

        inputs = processor(conversation=conversation, return_tensors="pt")
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = model.generate(**inputs, max_new_tokens=128)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f"{p_video_path}'s output caption: {response}")
        with open("video_llama3_output_log.jsonl", "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps({"video": p_video_path, "caption": response}, ensure_ascii=False) + "\n")
        captions.append(response)

    df['caption'] = captions

    df.to_csv("./video_llama3_evaluation.csv", index=False)


if __name__ == "__main__":
    import subprocess
    wait_time = 60 * 60 * 5
    print(f"The Program will start after {wait_time}s")
    time.sleep(wait_time)
    print("The Program has been started!")

    try:
        main()
    except Exception as e:
        print(f"❌ 프로그램 실행 중 오류 발생: {e}")
        print("⚠️ flash-attn 설치를 시도합니다...")
        subprocess.run(["pip", "install", "flash-attn", "--no-build-isolation"])
        print("📌 설치가 완료되었거나 실패했습니다. 수동으로 프로그램을 재시작해주세요.")