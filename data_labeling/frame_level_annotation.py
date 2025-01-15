import cv2
import os 
import requests
from PIL import Image
import json
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

def extract_frames(video_path , output_dir, video_name):
    """
    video(mp4) 경로에서 frame(.jpg) 로 자르는 함수

    Args : 
        video_path(str) : video 의 저장 경로
        output_dir(str) : 뽑은 frame을 저장할 디렉토리 경로
        video_name(str) : 해당 video 이름

    Returns:
        frame_info(list) : 해당 video에서 추출한 frames들의 순번(frame_numbe) , timestamp , frame_path 정보 반환

    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0 

    # video info
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS)
    print(f"total frame : {length} , Resolution :{width}x{height} , fps : {fps}")
    #frame 저장할 폴더 생성
    frames_dir = os.path.join(output_dir , video_name)
    frames_info=[]
    if not os.path.exists(frames_dir) :
        os.makedirs(frames_dir)

    while cap.isOpened():
        ret , frame = cap.read()
        if  not ret : 
            break
        timestamp = frame_count / fps
        frame_path = os.path.join(output_dir, video_name, f"frame_{frame_count:06d}.jpg")

        # BGR 을 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        frames_info.append({
            "frame_number": frame_count,
            "timestamp": timestamp,
            "frame_path": frame_path
        })
        
        frame_count+=1
    cap.release()
    return frames_info

def get_frames_captions_Blip2():
    """
    - video_data.json에 있는 모든 비디오에 대해 frame level annotation 생성
    flow
        (1) video_json에 있는 모든 비디오에 대해 frame 으로 자름 및 저장
        (2)  Blip2를 통해 frame 별 img_caption 생성 
        (3) "video_name", "frame_number" ,"timestamp" , frame_path" , "caption" 을 경로 ./frames/에 .json 형태로 저장

    """
    # Blip2 모델 설정
    Blip2_model_name = 'Salesforce/blip2-opt-2.7b'
    processor = AutoProcessor.from_pretrained(Blip2_model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(Blip2_model_name, torch_dtype=torch.float16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Video 목록 JSONL 파일 읽기
    with open('./video_data.json', 'r') as f:
        videos = [json.loads(line) for line in f]

    output_dir = "./frames"
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)
    
    # frame level annotation을 JSON 형태로 저장
    with open('./frame_annotations.json', 'w') as out_f:
        for video  in videos : 
            video_name = video['video_name']
            video_path = video['video_path']

            # 프레임 추출
            frames_info = extract_frames(video_path, output_dir, video_name)

            # 각 프레임에 대한 caption 생성
            for frame_info in frames_info: 
                frame_path = frame_info['frame_path']

                # 이미지 load 및 전처리
                image =  image = Image.open(frame_path).convert('RGB')  
                inputs = processor(image, return_tensors="pt").to(device, torch.float16)
                generated_ids = model.generate(**inputs, max_new_tokens=20)
                frame_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                # 결과 저장
                result = {
                    "video_name": video_name, # 참고한 video 이름 -> 이후 video id로 변경될 예정
                    "frame_number": frame_info['frame_number'], # 해당 frame의 video에서 순번
                    "timestamp": frame_info['timestamp'], # video 에서 frame 등장 시각
                    "frame_path": frame_path, # frame  저장 경로
                    "caption": frame_caption # frame의 image caption

                }

                # json 파일 저장
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                print(f"Finished!")


if __name__ == "__main__":
    get_frames_captions_Blip2()