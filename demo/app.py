import os
import numpy as np
import gradio as gr
import shutil
import ffmpeg
import subprocess
from database import run
# 시스템 경로를 추가하여 상위 경로 접근 가능하도록 변경
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.video_load import VideoLoad
captioner = VideoLoad(project_root)

def save_video(video: str) -> str:
    """
    비디오를 받아서 저장 후에 경로 반환

    args: 
    video (str): 저장할 비디오 경로
    
    returns:
      abs_path (str): 저장된 비디오 절대 경로
    """
    save_dir = "./data/tmp/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "output.mp4")
    shutil.copy(video, save_path)
    abs_path = os.path.abspath(save_path)

    return abs_path

import cv2
def extract_frame(video: str, timestamp: str) -> str:
    """
    비디오를 받아서 프레임 추출 후 저장 후에 경로 반환

    args: 
    video (str): 프레임 추출할 비디오 경로
    timestamp (str): 프레임 추출할 타임스탬프

    returns: 
    output_path (str): 추출된 프레임 절대 경로
    """
    video_path = save_video(video)
    output_path = os.path.abspath("./data/tmp/frame.jpg")
    
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path, frame)
    else:
        raise RuntimeError("지정한 시간에 프레임을 추출할 수 없습니다")
    
    cap.release()
    return output_path

def clipping_video(video: str, start: str, end: str) -> str:
    """
    args: 
    video (str): 클리핑할 비디오 경로
    start (str): 클리핑할 시작 타임스탬프
    end (str): 클리핑할 끝 타임스탬프

    returns: 
    clip_path (str): 클리핑된 비디오 절대 경로
    """
    video_path = save_video(video)
    clip_path = os.path.abspath("./data/tmp/clipped_video.mp4")
    os.makedirs(os.path.dirname(clip_path), exist_ok=True) 
    
    ffmpeg.input(video_path, ss=start, to=end).output(clip_path).run()
    
    if os.path.exists(video_path):
        os.remove(video_path)

    return clip_path

def update_video(video_path: str) -> tuple[gr.update, str]:
    """사용자가 입력한 video_path로 로컬 영상 파일 경로 업데이트"""
    if os.path.exists(video_path):
        return gr.update(value=video_path)
    else:
        print(f"❌ Error: {video_path} 파일이 존재하지 않습니다.")
        return gr.update(value=""), "Error: File not found"
    
    return gr.update(value="")


def update_image(image_path: str) -> tuple[gr.update, str]:
    """사용자가 입력한 image_path로 로컬 이미지 파일 경로 업데이트"""
    if os.path.exists(image_path):
        return gr.update(value=image_path)
    else:
        print(f"❌ Error: {image_path} 파일이 존재하지 않습니다.")
        return gr.update(value=""), "Error: File not found"

    return gr.update(value="")

data_dir = os.path.join(os.path.dirname(project_root), "data")
vid_idx_table = {'video1': "D3/DR/test/clips/D3_DR_0804_000048_002.mp4",
                 'video2': "D3/DR/test/clips/D3_DR_0908_000521_001.mp4",
                 'video3': "YT8M_test/MV/test/clips/yt8m_Movieclips_xcJXT5lc1Bg_001.mp4"}

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
    ):
    """
    비디오 아이디를 받아서 비디오를 반환
    """
    video_path = os.path.join(data_dir, vid_idx_table[video_id])

    video_path = clipping_video(video_path, timestamp_start, timestamp_end)

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
    ):
    """
    비디오 아이디를 받아서 비디오를 반환
    """
    video_path = os.path.join(data_dir, vid_idx_table[video_id])
    image_path = extract_frame(video_path, timestamp_start)
    media_tensor = captioner.load_media(
                        image_path,
                        media_type=media_type, 
                        num_segments=num_segments, 
                        resolution=resolution)

    caption = captioner.generate_caption(media_tensor, media_type, user_prompt, instruction)
    return update_image(image_path), caption

def text_capture_input(video, vocab):
    return f"Vocab: {vocab}"

video_interface = gr.Interface(
    fn=view_video,
    inputs=[gr.Textbox(label="Video ID"), gr.Textbox(label="Timestamp_start(HH:MM:SS)", placeholder="00:00:00"), 
            gr.Textbox(label="Timestamp_end(HH:MM:SS)", placeholder="00:00:00")
            ],
    outputs=[
                       gr.Video(label="Video"), gr.Textbox(label="Generated Caption")
                    ]
)

image_interface = gr.Interface(
    fn=view_image,
    inputs=[gr.Textbox(), gr.Textbox(label="Timestamp(HH:MM:SS)", placeholder="00:00:00")],
    outputs=[
                        gr.Image(label="Image"), gr.Textbox(label="Generated Caption")
                    ]
)

text_interface = gr.Interface(
    fn=text_capture_input,
    inputs=[gr.Video(), gr.Textbox(label="vocab")],
    outputs="image"
)


def video_capture_input(*videos):
    """동적 비디오 입력 처리 함수"""
    valid_videos = [v for v in videos if v is not None]
    return f"처리된 동영상 개수: {len(valid_videos)}"

def process_video_info(input_text):
    best_match, video_path = run(input_text)
    print(f"video_path: {video_path}")
    return best_match, update_video(video_path)

input_video_num = 3

multi_video_interface = gr.Interface(
    fn=video_capture_input,
    inputs=[gr.Video() for _ in range(input_video_num)],
    outputs="image"
)

with gr.Blocks() as demo:
    with gr.Tab("Video to Text"):
        with gr.Tabs():
            with gr.Tab("기본 평가"):
                image_interface.render()
            with gr.Tab("가산점 평가"):
                video_interface.render()
    with gr.Tab("Text to Video"):
        with gr.Tabs():
            with gr.Tab("기본 평가"):
                text_interface.render()
            with gr.Tab("가산점 평가"):
                with gr.Row(2) as colums:
                    # 왼쪽 컬럼: 입력 컨트롤
                    with gr.Column() as input_column:
                        gr.Markdown("### 비디오 입력 설정")
                        # 동영상 개수 입력 컴포넌트
                        video_count = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="동영상 입력 개수 선택"
                        )

                        # 비디오 입력을 위한 컨테이너
                        with gr.Row() as input_container:
                            input_text = gr.Textbox(label="검색어 입력", lines=2)
                            
                        
                        with gr.Row() as video_container:
                            video_inputs = []
                                
                            for i in range(10):
                                video_input = gr.Video(label=f"동영상 {i+1}", visible=False)
                                video_inputs.append(video_input)
                                
                        submit_btn = gr.Button("처리 시작", size='lg')
                    
                    # 오른쪽 컬럼: 출력 결과
                    with gr.Column() as output_column:
                        gr.Markdown("### 처리 결과")
                        output_text = gr.Textbox(label="처리된 비디오 정보", lines=2)
                        
                        with gr.Row():
                            output_image = gr.Image(label="결과 이미지")
                            output_video = gr.Video(label="결과 비디오")
                            submit_btn.click(fn=process_video_info, inputs=input_text, outputs=[output_text, output_video])
                        
            # 슬라이더 값 변경 시 비디오 입력 컴포넌트 표시/숨김 처리
            def update_visible_inputs(count):
                return [gr.update(visible=(i < count)) for i in range(10)]

            def update_retrieval_inputs_text(text):
                return gr.update(value=text)
            
            video_count.change(
                fn=update_visible_inputs,
                inputs=[video_count],
                outputs=video_inputs
            )
            # input_text.change(
            #     fn=update_retrieval_inputs_text,
            #     inputs=[input_text],
            #     outputs=input_text
            # )
            # 비디오 처리 함수
            def process_videos(*videos):
                valid_videos = [v for v in videos if v is not None]
                return f"처리된 동영상 개수: {len(valid_videos)}"

            # # 제출 버튼 클릭 시 처리
            # submit_btn.click(
            #     fn=process_videos,
            #     inputs=[*video_inputs, input_text],
            #     outputs=output_text
            # )



# def image_capture_input(video, video_id, timestamp):
#     video_id_var = video_id
#     timestamp_var = timestamp
#     caption = image_captioning(video, timestamp)
#     return caption





if __name__ == "__main__":
    demo.launch()
