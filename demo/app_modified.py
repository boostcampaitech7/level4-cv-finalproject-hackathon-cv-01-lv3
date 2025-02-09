import os
import numpy as np
import gradio as gr
import shutil
import ffmpeg
import subprocess
import cv2
from database import run
import asyncio
# 시스템 경로를 추가하여 상위 경로 접근 가능하도록 변경
import sys
from utils.InternVL_model import InternVL
from utils.InternVL_model import load_image
from deepl_trans import translate_deepl
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 절대 경로 지정
EXTERNAL_DIR = os.path.abspath("/data/ephemeral/home")
# YT8M은 yt8m + 외부 데이터 10개
YT8M_DIR = os.path.abspath("/data/ephemeral/home/YT8M")
os.makedirs(YT8M_DIR, exist_ok=True)
captioner = InternVL()



from translate import translation
from data.utils.clip_video import split_video_into_scenes
import json 

# 비디오 경로를 저장하기 위한 딕셔너리
vid_idx_table = {}

# 외부데이터를 위한 딕셔너리
external_idx_table = {}

def set_video_mapping():
    # 비디오 경로를 저장하기 위한 딕셔너리
    global vid_idx_table
    mapping_txt=  '/data/ephemeral/home/hanseonglee_demo/level4-cv-finalproject-hackathon-cv-01-lv3/demo/video_ids.txt'
    with open(mapping_txt, "r") as f:
        for line in f:
            video_id, video_path = line.strip().split(':')
            vid_idx_table[video_id] = video_path + ".mp4"
    return vid_idx_table


def save_video(video: str) -> str:
    """
    비디오를 받아서 저장 후에 경로 반환

    args: 
    video (str): 저장할 비디오 경로
    
    returns:
    abs_path (str): 저장된 비디오의 절대 경로
    """
    save_dir = EXTERNAL_DIR
    os.makedirs(save_dir, exist_ok=True)

    original_name = os.path.basename(video)  
    save_path = os.path.join(save_dir, original_name)  

    shutil.copy(video, save_path)
    # 원본 파일을 YT8M 디렉토리로 복사 TODO: external -> YT8M
    shutil.copy(video, "/data/ephemeral/home/YT8M/externals")
    abs_path = os.path.abspath(save_path)  

    return abs_path

def save_json_video(video_id: str, start_time: str, end_time: str, caption: str):
    """
    주어진 비디오 ID에 해당하는 비디오 파일명을 기반으로 JSON 파일을 생성하여 저장합니다.
    
    Args:
        video_id (str): 비디오의 고유 ID.
        start_time (str): 비디오 클립의 시작 시간 (HH:MM:SS).
        end_time (str): 비디오 클립의 종료 시간 (HH:MM:SS).
        caption (str): 해당 구간의 자동 생성된 캡션.
    
    Returns:
        gr.Info: JSON 저장 완료 메시지.
    """
    video_name = os.path.splitext(os.path.basename(vid_idx_table[video_id]))[0]
    
    json_data = {
        video_name: {
            "start_time": start_time,
            "end_time": end_time,
            "caption": caption
        }
    }
    
    json_path = os.path.join("./data", f"{video_name}.json")

    # JSON 파일 저장
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)
    
    return gr.Info(f"✅ JSON 파일이 저장되었습니다: {json_path}")


def save_json_image(video_id: str, timestamp: str, caption: str):
    """
    주어진 비디오 ID에 해당하는 비디오 파일명을 기반으로 JSON 파일을 생성하여 저장.
    
    Args:
        video_id (str): 비디오의 고유 ID.
        timestamp (str): 특정 프레임의 타임스탬프 (HH:MM:SS).
        caption (str): 해당 프레임의 자동 생성된 캡션.
    
    Returns:
        gr.Info: JSON 저장 완료 메시지.
    """
    image_name = os.path.splitext(os.path.basename(vid_idx_table[video_id]))[0]
    
    json_data = {
        image_name: {
            "timestamp": timestamp,
            "caption": caption
        }
    }
    
    json_path = os.path.join("./data", f"{image_name}.json")

    # JSON 파일 저장
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)
    
    return gr.Info(f"✅ JSON 파일이 저장되었습니다: {json_path}")


def extract_frame(video_path: str, timestamp: str) -> str:
    """
    비디오를 받아서 프레임 추출 후 저장 후에 경로 반환

    args: 
    video_path (str): 프레임 추출할 비디오 경로
    timestamp (str): 프레임 추출할 타임스탬프

    returns: 
    output_path (str): 추출된 프레임 절대 경로
    """
    os.makedirs("./data/tmp/", exist_ok=True)
    output_path = os.path.join("./data/tmp/", os.path.basename(video_path).replace(".mp4", "_frame.jpg"))
    


    # 타임스탬프를 초 단위로 변환
    h, m, s = map(int, timestamp.split(':'))
    total_seconds = h * 3600 + m * 60 + s
    
    # OpenCV를 이용한 프레임 추출
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("비디오 파일을 열 수 없습니다")
    
    # 밀리초 단위로 타임스탬프 설정
    cap.set(cv2.CAP_PROP_POS_MSEC, total_seconds * 1000)
    success, frame = cap.read()
    
    if success:
        cv2.imwrite(output_path, frame)
    else:
        raise RuntimeError("지정한 시간에 프레임을 추출할 수 없습니다")
    
    cap.release()
    return output_path

def save_json_clip(origin_name: str, start_time: str, end_time: str):
    """
    주어진 비디오 ID에 해당하는 비디오 파일명을 기반으로 JSON 파일을 생성하여 저장합니다.
    
    Args:
        video_id (str): 비디오의 고유 ID.
        start_time (str): 비디오 클립의 시작 시간 (HH:MM:SS).
        end_time (str): 비디오 클립의 종료 시간 (HH:MM:SS).
    
    Returns:
        None
    """
    # video_name = os.path.splitext(os.path.basename(vid_idx_table[video_id]))[0]
    video_name = origin_name
    json_data = {
        video_name: {
            "start_time": start_time,
            "end_time": end_time,
            "caption": ""
        }
    }
    
    json_path = os.path.join("./data", f"{video_name}.json")
    # JSON 파일 저장``
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)
    
    return json_path


def clipping_video(video_path: str, start: str, end: str) -> str:
    """
    args: 
    video (str): 클리핑할 비디오 경로
    start (str): 클리핑할 시작 타임스탬프
    end (str): 클리핑할 끝 타임스탬프

    returns: 
    clip_path (str): 클리핑된 비디오 절대 경로
    """
    clip_path = os.path.abspath(f"{os.path.basename(video_path)}_clipped.mp4")
    os.makedirs(os.path.dirname(clip_path), exist_ok=True) 
    
    if os.path.exists(clip_path):
        os.remove(clip_path)
    
    ffmpeg.input(video_path, ss=start, to=end).output(
        clip_path, 
        # vcodec='libx264', 
        # crf=23, 
        # preset='fast', 
        # acodec='aac', 
        # audio_bitrate='128k'
    ).run()
    return clip_path   

def update_video(video_path: str) -> dict:
    """
    video_path로 로컬 영상 파일 경로 업데이트
    
    args: 
    video_path(str): 비디오 경로
    
    returns:
    gr.update(value="video_path"): 해당 경로가 존재하면 업데이트
    """
    if os.path.exists(video_path):
        return gr.update(value=video_path)
    else:
        print(f"Error: {video_path} 파일이 존재하지 않습니다.")
        return gr.update(value="")  

def update_image(image_path: str) -> dict:
    """
    image_path로 로컬 이미지 파일 경로 업데이트
    
    args:
    image_path(str):이미지 경로
    
    returns: 
    gr.update(value="image_path"): 해당 경로가 존재하면 업데이트
    """
    if os.path.exists(image_path):
        return gr.update(value=image_path)
    else:
        print(f"Error: {image_path} 파일이 존재하지 않습니다.")
        return gr.update(value="")  

data_dir = os.path.join(os.path.dirname(project_root), "data")
print(data_dir)


def view_video(
        video_id = None,
        timestamp_start = 0,
        timestamp_end = 0,
        media_type = 'video',
        num_segments = 8,
        resolution = 224,
        hd_num = 6
    ) -> tuple:
    """
    비디오 아이디를 받아서 비디오를 반환
    
    args:
    video_id(str),
    timestamp_start(int): 구간 시작 지점,
    timestamp_end(int): 구간 끝 지점,
    media_type(str): image인지 video인지를 명시,
    num_segments(int),
    resolution(int),
    hd_num(int): HD Frame 수,
    
    returns:
    video_path(str): 비디오 경로,
    caption(str): 생성된 캡션
    """
    video_path = os.path.join(YT8M_DIR,"movieclips", vid_idx_table[video_id])

    video_path = clipping_video(video_path, timestamp_start, timestamp_end)
    vision_json_path = save_json_clip(video_path, timestamp_start, timestamp_end)

    print(f"\n\n[DEBUG] video_path: {video_path}")
    print(f"[DEBUG] vision_json_path: {vision_json_path}")

    #TODO: 경로 수정 필요
    # base_name "/data/ephemeral/home/yt8m_Movieclips__7WnJtSpIP4.mp4" -> "yt8m_Movieclips__7WnJtSpIP4/tmp"
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n\n[DEBUG] base_name: {base_name}")
    tmp_dir = os.path.join(base_name, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"\n\n[DEBUG] tmp_dir: {tmp_dir}")



    # vision_json_path = os.path.join(data_dir, 'YT8M', 'Movieclips', 'test', 'labels', f'{video_id}.json')
    # base_name = '_'.join(video_id.split('_')[:-1])
    # speech_json_path = os.path.join(data_dir, 'YT8M', 'Movieclips', 'test', 'stt', f'{base_name}.json')
    # summary_dir = os.path.join(data_dir, 'YT8M', 'Movieclips', 'test', 'summary')
    # video_name = video_id
    # TODO: video captioning 필요
    stt_dir = os.path.join(tmp_dir, "stt")
    summary_dir = os.path.join(tmp_dir, "summary")
    os.makedirs(stt_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    try:
        stt_result = send_for_stt(video_path)
        stt_path = os.path.join(stt_dir, f"{base_name}.json")
        with open(stt_path, 'w') as f:
            json.dump(stt_result, f)
    except Exception as e:
        print(f"STT 처리 실패: {str(e)}")
        
    # 3. Summary 생성
    try:
        summary_result = send_for_summary(video_path)
        summary_path = os.path.join(summary_dir, f"{base_name}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_result, f)
    except Exception as e:
        print(f"요약 생성 실패: {str(e)}")


    pixel_values, num_patches_list = InternVL.load_single_video(video_path)
    caption = captioner.generate_caption(
        media_tensor=(pixel_values, num_patches_list),
        media_type='video',
        #label 위치
        vision_caption_json_path=vision_json_path,
        speech_caption_json_path=stt_path,
        summary_dir=summary_dir,
        video_name=base_name
    )
    ko_caption = translate_deepl(caption, 'en', 'ko')
    return update_video(video_path), ko_caption

def process_video_info(input_text):
    """
    input_text: 검색어
    """
    input_text = translate_deepl(input_text, 'ko', 'en')
    best_match, segment_name = run(input_text)
    print(f"\n\n[DEBUG] best_match: {best_match}")
    print(f"[DEBUG] segment_name: {segment_name}")
    # yt8m_Movieclips_le6AAhqa_8U_001 -> yt8m_Movieclips_le6AAhqa_8U
    base_name = '_'.join(segment_name.split('_')[:-1])
    print(f"\n\n[DEBUG] base_name: {base_name}")
    # base_name + ".mp4"
    video_name = base_name + ".mp4"
    video_path = os.path.join(YT8M_DIR, "movieclips", video_name)
    print(f"\n\n[DEBUG] video_path: {video_path}")

    _, start_time, end_time, _ = best_match.split(',')
    timestamp = start_time[-8:] + " ~ " + end_time[-8:]  # HH:MM:SS 형식
    return best_match, timestamp, update_video(video_path)

def view_image(
        video_id = None,
        timestamp_start = 0,
        media_type = 'image',
        resolution = 224,   
        num_segments = 1
    ) -> tuple:
    """
    이미지 아이디를 받아서 이미지를 반환
    
    args:
    video_id(str),
    timestamp_start(int): 구간 시작 지점,
    timestamp_end(int): 구간 끝 지점,
    media_type(str): image인지 video인지를 명시,
    num_segments(int),
    resolution(int),
    hd_num(int): HD Frame 수,
    
    returns:
    image_path(str): 비디오 경로,
    caption(str): 생성된 캡션
    """
    video_path = os.path.join(YT8M_DIR,"movieclips", vid_idx_table[video_id])

    print(f"video_path: {video_path}")
    #TODO: 경로 수정 필요
    print(f"video_id: {video_id}")
    image_path = extract_frame(video_path, timestamp_start)
    print(image_path)

    #TODO: 경로 수정 필요   
    # vision_json_path = os.path.join(data_dir, 'YT8M', 'Movieclips', 'test', 'labels', f'{video_id}.json')
    # base_name = '_'.join(video_id.split('_')[:-1])
    # speech_json_path = os.path.join(data_dir, 'YT8M', 'Movieclips', 'test', 'stt', f'{base_name}.json')
    # summary_dir = os.path.join(data_dir, 'YT8M', 'Movieclips', 'test', 'summary_json')
    # video_name = video_id


    media_tensor = load_image(image_path)
    caption = captioner.generate_caption_image(
        media_tensor=media_tensor,
        media_type="image"
    )
    ko_caption = translate_deepl(caption, 'en', 'ko')
    return update_image(image_path), ko_caption




def process_base_info(*args:str) -> tuple:
    """
    T2V 기본 평가를 위한 비디오 처리 함수
    
    args(str): 비디오 경로(10개), 한국어 검색어
    
    returns:
    video_id(str): 비디오 아이디,
    timestamp(str): 비디오 구간,
    update_video(dict): 비디오 경로 업데이트
    """
    input_videos = args[:-1]
    # 한국어로 번역
    input_text = asyncio.run(translation(args[-1], 'ko'))
    print(input_text)
    valid_videos = [v for v in input_videos if v is not None]
    print(f"처리된 동영상 개수: {len(valid_videos)}")

    best_match, video_path = run(input_text)
    print(f"video_path: {video_path}")
    
    video_id, start_time, end_time, _ = best_match.split(',')
    video_id=video_id[-15:-4]
    timestamp = start_time[-8:]+" ~ "+end_time[-8:]

    return video_id, timestamp, update_video(video_path)



def download_video(*videos:str) :
    global vid_idx_table
    """
    비디오를 다운로드하여 저장.
    저장된 파일 목록을 반환.
    
    args:
    videos(str): 비디오 경로(10개)
    
    returns:
    gr.Info(-> Union[str, gr.components.Info]): 동영상 제출 관련 메시지
    """
    
    for i, video in enumerate(videos):
        if not video:
            continue  
        
        if isinstance(video, dict):
            video_path = video.get("name")
        else:
            video_path = video
        
        if not video_path or not os.path.exists(video_path):
            print(f"⚠️ Invalid video path: {video_path}")
            continue  

        vid_idx_table[f"external_{i+1}"] = save_video(video_path)
        external_idx_table[f"external_{i+1}"] = video_path

    # 1. Trimming 및 세그먼트 분할
    
    # 경로 설정
    base_dir = EXTERNAL_DIR

    base_dir = os.path.join(base_dir, "data", "download_video")
    clip_dir = os.path.join(base_dir,'test',"clips")
    label_dir = os.path.join(base_dir, "test", "labels")
    stt_dir = os.path.join(base_dir, "test","stt")
    summary_dir = os.path.join(base_dir,"test" , "summary")
    
    # 디렉토리 생성
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(stt_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    # 각 비디오 처리 파이프라인
    for vid_key, video_path in external_idx_table.items():
        # 1. Scene 분할
        split_video_into_scenes(
            video_path,
            output_json_dir=label_dir,
            segments_dir=clip_dir
        )
        
        # 2. STT 처리
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        try:
            stt_result = send_for_stt(video_path)
            stt_path = os.path.join(stt_dir, f"{base_name}.json")
            with open(stt_path, 'w') as f:
                json.dump(stt_result, f)
        except Exception as e:
            print(f"STT 처리 실패: {str(e)}")
            
        # 3. Summary 생성
        try:
            summary_result = send_for_summary(video_path)
            summary_path = os.path.join(summary_dir, f"{base_name}.json")
            with open(summary_path, 'w') as f:
                json.dump(summary_result, f)
        except Exception as e:
            print(f"요약 생성 실패: {str(e)}")
        gr.Info("✅ summary 생성 완료")
        # 4. 캡션 생성 (수정 필요)
        # base_dir의 절대 경로
        absolute_base_dir = EXTERNAL_DIR
        clip_files = [f for f in os.listdir(clip_dir) if f.endswith('.mp4')]
        for clip_file in clip_files:
            clip_path = os.path.join(clip_dir, clip_file)
            pixel_values, num_patches_list = InternVL.load_single_video(clip_path)
            # 경로 자동 생성
            video_id = os.path.splitext(clip_file)[0]
            base_name = '_'.join(video_id.split('_')[:-1])
            vision_json_path = os.path.join(label_dir, f"{video_id}.json")
            speech_json_path = os.path.join(stt_dir, f"{base_name}.json")
            
            caption = captioner.generate_caption(
                media_tensor=(pixel_values, num_patches_list),
                media_type='video',
                vision_caption_json_path=vision_json_path,
                speech_caption_json_path=speech_json_path,
                summary_dir=summary_dir,
                video_name=base_name
            )
            print(f"VISION_JSON_PATH: {vision_json_path}")
            print(f"CAPTION: {caption}")
            # label json에 캡션 추가
            with open(vision_json_path, 'r') as f:
                data = json.load(f)
                data['caption'] = caption
            with open(vision_json_path, 'w') as f:
                json.dump(data, f)

            gr.Info("✅ caption 생성 완료")
            print(f"video_id: {video_id}, base_name: {base_name}, vision_json_path: {vision_json_path}, speech_json_path: {speech_json_path}, summary_dir: {summary_dir}, caption: {caption}")


    if external_idx_table:
        file_list_str = "\n".join([f"{key}: {os.path.basename(value)}" for key, value in external_idx_table.items()])
        gr.Info("✅ 동영상 제출이 완료되었습니다!")
        return file_list_str
    else:
        gr.Info("⚠️ 제출할 동영상이 없습니다!")
        return ""
    

def clear_videos():
    """
    SAVE_DIR에 저장된 동영상을 모두 삭제
    
    returns:
    gr.Info(-> Union[str, gr.components.Info]): 동영상 삭제 관련 메시지
    """
    global vid_idx_table
    SAVE_DIR = "./data/download_video"
    if os.path.exists(SAVE_DIR):
        for file in os.listdir(SAVE_DIR):
            file_path = os.path.join(SAVE_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                vid_idx_table.clear()
        return gr.Info("🗑️ 저장된 동영상이 모두 삭제되었습니다!")
    else:
        return gr.Info("⚠️ 삭제할 동영상이 없습니다.")


###### Interface ######

# 비디오 인터페이스
video_interface = gr.Interface(
    fn=view_video,
    # 비디오 입력
    inputs=[gr.Textbox(label="Video ID"), gr.Textbox(label="Timestamp_start(HH:MM:SS)", placeholder="00:00:00"), 
            gr.Textbox(label="Timestamp_end(HH:MM:SS)", placeholder="00:00:00")
            ],
    # 비디오 출력
    outputs=[
                       gr.Video(label="Video"), gr.Textbox(label="Generated Caption")
                    ]
)

# 이미지 인터페이스
image_interface = gr.Interface(
    fn=view_image,
    # 이미지 입력
    inputs=[gr.Textbox(), gr.Textbox(label="Timestamp(HH:MM:SS)", placeholder="00:00:00")],
    # 이미지 출력
    outputs=[
                        gr.Image(label="Image"), gr.Textbox(label="Generated Caption")
                    ]
)

###### demo ######
with gr.Blocks() as demo:
    file_list_output = gr.Textbox(label="업로드한 파일 목록", lines=10)
    # 비디오 입력 탭
    with gr.Tab("비디오 입력"):
        video_inputs = []
        with gr.Column():
            for i in range(0, 10, 2):
                with gr.Row():
                    video_inputs.append(gr.Video(label=f"동영상 {i+1}"))
                    video_inputs.append(gr.Video(label=f"동영상 {i+2}"))                    
        
        with gr.Row(2):
            with gr.Column():
                submit_btn = gr.Button("제출", size='lg')
            with gr.Column():
                clear_btn = gr.Button("삭제", size='lg')
        
        # 제출 버튼 클릭 시, 업로드된 비디오를 다운로드
        submit_btn.click(fn=download_video, inputs=[*video_inputs], outputs=[file_list_output])
        # 삭제 버튼 클릭 시, 업로드된 비디오 삭제
        clear_btn.click(fn=clear_videos, outputs=[file_list_output])
 # V2T 탭    
    with gr.Tab("Video to Text"):
        with gr.Tabs():
            with gr.Tab("기본 평가"):
                with gr.Row(2) as columns:
                    with gr.Column() as input_column:
                        video_id_input = gr.Textbox(label="Video ID")
                        timestamp_input = gr.Textbox(label="Timestamp(HH:MM:SS)", placeholder="00:00:00")
                        with gr.Row(2):
                            clear_btn = gr.Button("Clear", size='lg')
                            submit_btn = gr.Button("submit", size='lg')
                    with gr.Column() as output_column:
                        image_output = gr.Image(label="Generated Image")
                        caption_output = gr.Textbox(label="Generated Caption")

                        json_btn = gr.Button("JSON으로 저장", size='lg')
                        json_output = gr.Info()
                        
                    # 클리어 버튼 클릭 시, 입력 초기화
                    clear_btn.click(
                        fn=lambda: ("", "", None, ""),  # 빈 값 반환 (이미지는 None)
                        inputs=[],
                        outputs=[video_id_input, timestamp_input, image_output, caption_output]
                    )
                    
                    # 제출 버튼 클릭 이벤트
                    submit_btn.click(
                        fn=view_image, 
                        inputs=[video_id_input, timestamp_input], 
                        outputs=[image_output, caption_output]
                    )
                    
                    # JSON 저장 버튼 클릭 이벤트
                    json_btn.click(
                        fn=save_json_image, 
                        inputs=[video_id_input, timestamp_input, caption_output], 
                        outputs=json_output
                    )
            with gr.Tab("가산점 평가"):
                with gr.Row(2) as columns:
                    with gr.Column() as input_column:
                        video_id_input = gr.Textbox(label="Video ID")
                        start_time_input = gr.Textbox(label="Start time(HH:MM:SS)", placeholder="00:00:00")
                        end_time_input = gr.Textbox(label="End time(HH:MM:SS)", placeholder="00:00:00")
                        with gr.Row(2):
                            clear_btn = gr.Button("Clear", size='lg')
                            submit_btn = gr.Button("submit", size='lg')
                    with gr.Column() as output_column:
                        video_output = gr.Video(label="Generated Clip")
                        caption_output = gr.Textbox(label="Generated Caption")

                        json_btn = gr.Button("JSON으로 저장", size='lg')
                        json_output = gr.Info()
                        
                    # 클리어 버튼 클릭 시, 입력 초기화
                    clear_btn.click(
                        fn=lambda: ("", "", None, ""),  # 빈 값 반환 (이미지는 None)
                        inputs=[],
                        outputs=[video_id_input, start_time_input, end_time_input, caption_output]
                    )
                    
                    # 제출 버튼 클릭 이벤트
                    submit_btn.click(
                        fn=view_video, 
                        inputs=[video_id_input, start_time_input, end_time_input], 
                        outputs=[video_output, caption_output]
                    )
                    
                    # JSON 저장 버튼 클릭 이벤트
                    json_btn.click(
                        fn=save_json_video, 
                        inputs=[video_id_input, start_time_input, end_time_input, caption_output], 
                        outputs=json_output
                    )
    # T2V 탭
    with gr.Tab("Text to Video"):
        with gr.Tabs():
            # with gr.Tab("기본 평가"):
            #     with gr.Row(2) as columns:
            #         with gr.Column() as input_column:
            #             gr.Markdown("### 비디오 입력 설정")
            #             input_text = gr.Textbox(label="검색어 입력", lines=2)
            #             submit_btn = gr.Button("처리 시작", size='lg')

            #         with gr.Column():
            #             gr.Markdown("### 처리 결과")
            #             output_id = gr.Textbox(label="Video ID")
            #             output_timestamp = gr.Textbox(label="Timestamp")
            #             output_video = gr.Video(label="결과 비디오")
            #     # 처리 시작 버튼 클릭 시, 활성화된 비디오만 입력으로 사용
            #     submit_btn.click(fn=process_base_info, inputs=[*video_inputs, input_text], outputs=[output_id, output_timestamp, output_video])
                
            with gr.Tab("가산점 평가"):
                with gr.Row(2) as columns:
                    # 왼쪽 컬럼: 입력 컨트롤
                    with gr.Column() as input_column:
                        gr.Markdown("### 비디오 입력 설정")

                        input_text = gr.Textbox(label="검색어 입력", lines=2)

                        submit_btn = gr.Button("처리 시작", size='lg')

                    # 오른쪽 컬럼: 출력 컨트롤
                    with gr.Column():
                        gr.Markdown("### 처리 결과")
                        output_id = gr.Textbox(label="Video_id")
                        output_timestamp = gr.Textbox(label="Timestamp")
                        output_video = gr.Video(label="결과 비디오")
                # 처리 시작 버튼 클릭 시, 활성화된 비디오만 입력으로 사용
                submit_btn.click(fn=process_video_info, inputs=input_text, outputs=[output_id, output_timestamp, output_video])

                        
            # 비디오 처리 함수
            def process_videos(*videos):
                valid_videos = [v for v in videos if v is not None]
                return f"처리된 동영상 개수: {len(valid_videos)}"

# 외부 서버 요청을 위한 헬퍼 함수 (구현 필요)
import requests
from typing import Dict, Any

def send_for_stt(video_path: str) -> Dict[str, Any]:
    """STT 서버로 비디오 전송 (10.28.224.152:30172)"""
    url = "http://10.28.224.152:30172/whisper"
    
    with open(video_path, 'rb') as video_file:
        files = {'video': (os.path.basename(video_path), video_file, 'video/mp4')}
        response = requests.post(url, files=files, timeout=30)
    
    if response.status_code != 200:
        raise Exception(f"STT 서버 오류: {response.text}")
    
    return response.json()

def send_for_summary(video_path: str) -> Dict[str, Any]:
    """요약 서버로 비디오 전송 (구현 필요)"""
    # 실제 요약 서버 엔드포인트로 변경 필요
    url = "http://10.28.224.170:30170/summerize"
    
    try:
        with open(video_path, 'rb') as video_file:
            files = {'video': (os.path.basename(video_path), video_file, 'video/mp4')}
            response = requests.post(url, files=files, timeout=30)
        
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"요약 생성 실패: {str(e)}")

if __name__ == "__main__":
    set_video_mapping()
    demo.launch(
        debug=True,       # 디버그 모드 활성화 (에러 상세 출력)
        show_error=True,  # UI에 에러 직접 표시
    )
