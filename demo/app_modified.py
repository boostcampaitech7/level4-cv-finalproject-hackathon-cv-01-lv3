import os
import json
import numpy as np
import gradio as gr
import shutil
import ffmpeg
import subprocess
from database import run
import asyncio
import cv2

# 시스템 경로를 추가하여 상위 경로 접근 가능하도록 변경
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.video_load import VideoLoad
captioner = VideoLoad(project_root)

from translate import translation

# 비디오 경로를 저장하기 위한 딕셔너리
vid_idx_table = {}

def save_video(video: str) -> str:
    """
    비디오를 받아서 저장 후에 경로 반환

    args: 
    video (str): 저장할 비디오 경로
    
    returns:
    abs_path (str): 저장된 비디오의 절대 경로
    """
    save_dir = "./data/download_video/"
    os.makedirs(save_dir, exist_ok=True)

    original_name = os.path.basename(video)  
    save_path = os.path.join(save_dir, original_name)  

    shutil.copy(video, save_path)  
    abs_path = os.path.abspath(save_path)  

    return abs_path


def extract_frame(video_path: str, timestamp: str) -> str:
    """
    비디오를 받아서 프레임 추출 후 저장 후에 경로 반환

    args: 
    video_path (str): 프레임 추출할 비디오 경로
    timestamp (str): 프레임 추출할 타임스탬프

    returns: 
    output_path (str): 추출된 프레임 절대 경로
    """
    # 비디오 파일 이름 추출 (확장자 제거)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 프레임 저장 경로
    output_path = os.path.abspath(f"./data/tmp/{video_name}_frame.jpg")
    
    h, m, s = map(int, timestamp.split(':'))
    total_seconds = h * 3600 + m * 60 + s
    
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


def clipping_video(video_path: str, start: str, end: str) -> str:
    """
    args: 
    video (str): 클리핑할 비디오 경로
    start (str): 클리핑할 시작 타임스탬프
    end (str): 클리핑할 끝 타임스탬프

    returns: 
    clip_path (str): 클리핑된 비디오 절대 경로
    """
    
    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    
    clip_path = os.path.abspath(f"./data/tmp/{clip_name}_clipped_video.mp4")
    
    os.makedirs(os.path.dirname(clip_path), exist_ok=True) 
    
    if os.path.exists(clip_path):
        os.remove(clip_path)
    
    ffmpeg.input(video_path, ss=start, to=end).output(
        clip_path, 
        vcodec='libx264', 
        crf=23, 
        preset='fast', 
        acodec='aac', 
        audio_bitrate='128k'
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
        hd_num = 6,
        user_prompt = "Describe the video step by step",
        instruction = "Carefully watch the video and describe what is happening in detail"
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
    user_prompt(str),
    instruction(str)
    
    returns:
    video_path(str): 비디오 경로,
    caption(str): 생성된 캡션
    """
    video_path = os.path.join(data_dir, vid_idx_table[video_id])

    video_path = clipping_video(video_path, timestamp_start, timestamp_end)
    print(video_path)

    media_tensor = captioner.load_media(
                            video_path,
                            media_type=media_type, 
                            num_segments=num_segments, 
                            resolution=resolution, 
                            hd_num=hd_num)

    caption = captioner.generate_caption(media_tensor, media_type, user_prompt, instruction)
    return update_video(video_path), caption


def view_image(
        video_id = None,
        timestamp_start = 0,
        media_type = 'image',
        resolution = 224,
        num_segments = 1,
        user_prompt = "Describe the image step by step",
        instruction = "Carefully watch the image and describe what is happening in detail"
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
    user_prompt(str),
    instruction(str)
    
    returns:
    image_path(str): 비디오 경로,
    caption(str): 생성된 캡션
    """
    video_path = os.path.join(data_dir, vid_idx_table[video_id])
    image_path = extract_frame(video_path, timestamp_start)
    print(image_path)
    media_tensor = captioner.load_media(
                        image_path,
                        media_type=media_type, 
                        num_segments=num_segments, 
                        resolution=resolution)

    caption = captioner.generate_caption(media_tensor, media_type, user_prompt, instruction)
    return update_image(image_path), caption


def process_video_info(*args:str) -> tuple:
    """
    T2V 가산점 평가를 위한 비디오 처리 함수
    
    args(str): 비디오 경로(10개), 한국어 검색어
    
    returns:
    video_id(str): 비디오 아이디,
    timestamp(str): 비디오 구간,
    update_video(dict): 비디오 경로 업데이트
    """
    input_videos = args[:-1]
    # 한국어로 번역
    input_text = asyncio.run(translation(args[-1],'ko'))

    valid_videos = [v for v in input_videos if v is not None]
    print(f"처리된 동영상 개수: {len(valid_videos)}")

    best_match, video_path = run(input_text)
    print(f"video_path: {video_path}")

    video_id, start_time, end_time, _ = best_match.split(',')
    video_id=video_id[-15:-4]
    timestamp = start_time[-8:]+" ~ "+end_time[-8:]
    
    return video_id, timestamp, update_video(video_path)


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



def download_video(*videos:str):
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

        vid_idx_table[f"video{i+1}"] = save_video(video_path)

    if vid_idx_table:
        file_list_str = "\n".join([f"{key}: {os.path.basename(value)}" for key, value in vid_idx_table.items()])
        gr.Info("✅ 동영상 제출이 완료되었습니다!")
        return file_list_str
    else:
        gr.Info("⚠️ 제출할 동영상이 없습니다!")
        return ""


def clear_videos():
    """
    SAVE_DIR에 저장된 동영상을 모두 삭제
    
    args: None
    
    returns:
    gr.Info(-> Union[str, gr.components.Info]): 동영상 삭제 관련 메시지
    """
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


def save_json_video(video_id: str, start_time: str, end_time: str, caption: str):
    video_name=os.path.splitext(os.path.basename(vid_idx_table[video_id]))[0]
    
    json_data = {
        video_name: {
            "start_time": start_time,
            "end_time": end_time,
            "caption": caption
        }
    }
    
    json_path = os.path.join("./data", f"{video_name}.json")
    
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)
    
    return gr.Info(f"✅ JSON 파일이 저장되었습니다.")

def save_json_image(video_id: str, timestamp:str, caption: str):
    image_name=os.path.splitext(os.path.basename(vid_idx_table[video_id]))[0]
    
    json_data = {
        image_name: {
            "timestamp": timestamp,
            "caption": caption
        }
    }
    
    json_path = os.path.join("./data", f"{image_name}.json")
    
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)
    
    return gr.Info(f"✅ JSON 파일이 저장되었습니다.")


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
            with gr.Tab("기본 평가"):
                with gr.Row(2) as columns:
                    with gr.Column() as input_column:
                        gr.Markdown("### 비디오 입력 설정")
                        input_text = gr.Textbox(label="검색어 입력", lines=2)
                        submit_btn = gr.Button("처리 시작", size='lg')

                    with gr.Column():
                        gr.Markdown("### 처리 결과")
                        output_id = gr.Textbox(label="Video ID")
                        output_timestamp = gr.Textbox(label="Timestamp")
                        output_video = gr.Video(label="결과 비디오")
                # 처리 시작 버튼 클릭 시, 활성화된 비디오만 입력으로 사용
                submit_btn.click(fn=process_base_info, inputs=[*video_inputs, input_text], outputs=[output_id, output_timestamp, output_video])
                
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
                submit_btn.click(fn=process_video_info, inputs=[*video_inputs, input_text], outputs=[output_id, output_timestamp, output_video])

                        
            # 비디오 처리 함수
            def process_videos(*videos):
                valid_videos = [v for v in videos if v is not None]
                return f"처리된 동영상 개수: {len(valid_videos)}"


if __name__ == "__main__":
    demo.launch()
