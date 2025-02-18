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
from config import ServiceConfig  # 추가된 임포트

# 절대 경로 지정
EXTERNAL_DIR = os.path.abspath("/data/ephemeral/home")
# YT8M은 yt8m + 외부 데이터 10개
YT8M_DIR = os.path.abspath("/data/ephemeral/home/YT8M")
os.makedirs(YT8M_DIR, exist_ok=True)
captioner = InternVL()



from translate import translation
from data.utils.clip_video import split_video_into_scenes, spllit_video_ffmpeg
import json 
from database import get_elasticsearch_client, VideoCaption
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
    shutil.copy(video, "/data/ephemeral/home/YT8M/Movieclips")
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
    
    def time_to_seconds(t):
        h, m, s = map(int, t.split(':'))
        return h * 3600 + m * 60 + s
    
    start_seconds = time_to_seconds(start)
    end_seconds = time_to_seconds(end)

    if end_seconds <= start_seconds:
        end_seconds = start_seconds + 1  # 최소 1초 클립 보장
    ffmpeg.input(video_path, ss=start_seconds, to=end_seconds).output(
        clip_path,
        vcodec='copy',  # 원본 비디오 스트림 복사
        **{
            'c:a': 'aac',  # 오디오만 재인코딩
            'b:a': '128k'
        },
        strict='normal'
    ).run(overwrite_output=True)
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



def view_video(
        video_id = None,
        timestamp_start = 0,
        timestamp_end = 0,
    ) -> tuple:
    """
    비디오 아이디를 받아서 비디오를 반환
    
    args:
    video_id(str),
    timestamp_start(int): 구간 시작 지점,
    timestamp_end(int): 구간 끝 지점,
    
    returns:
    video_path(str): 비디오 경로,
    caption(str): 생성된 캡션
    """
    video_path = os.path.join(YT8M_DIR,"movieclips", vid_idx_table[video_id])
    video_path = clipping_video(video_path, timestamp_start, timestamp_end)
    vision_json_path = save_json_clip(video_path, timestamp_start, timestamp_end)


    base_name = os.path.splitext(os.path.basename(video_path))[0]
    base_name = base_name.replace(".mp4", "")
    # base_name = yt8m_Movieclips_QNIla2oQuBE_clipped
    tmp_dir = os.path.join(base_name, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)



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
        
    # Summary 생성
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
    # yt8m_Movieclips_le6AAhqa_8U_001 -> yt8m_Movieclips_le6AAhqa_8U
    base_name = '_'.join(segment_name.split('_')[:-1])
    # base_name + ".mp4"
    video_name = base_name + ".mp4"
    video_path = os.path.join(YT8M_DIR, "movieclips", video_name)

    video_id, start_time, end_time, _ = best_match.split(',')


    start_time = start_time[-8:]
    end_time = end_time[-8:]
    video_path = clipping_video(video_path, start_time, end_time)
    timestamp = start_time + " ~ " + end_time  # HH:MM:SS 형식
    return video_id, timestamp, update_video(video_path)

def view_image(
        video_id = None,
        timestamp_start = 0,
    ) -> tuple:
    """
    이미지 아이디를 받아서 이미지를 반환
    
    args:
    video_id(str),
    timestamp_start(int): 구간 시작 지점,
    
    returns:
    image_path(str): 비디오 경로,
    caption(str): 생성된 캡션
    """
    video_path = os.path.join(YT8M_DIR,"movieclips", vid_idx_table[video_id])

    print(f"video_path: {video_path}")
    print(f"video_id: {video_id}")
    image_path = extract_frame(video_path, timestamp_start)
    print(image_path)



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
    
    args:s
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
        print(f"\n\n!!!!!!!!!!!!!!!![DEBUG] video_path: {video_path}")
        if not video_path or not os.path.exists(video_path):
            print(f"⚠️ Invalid video path: {video_path}")
            continue  

        vid_idx_table[f"external{i+1}"] = save_video(video_path)
        external_idx_table[f"external{i+1}"] = video_path

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
    for vid_key, video_path in external_idx_table.items():
        spllit_video_ffmpeg(
            video_path,
            output_json_dir=label_dir,
            segments_dir=clip_dir
        )
        # Scene Detect 사용 시
        # split_video_into_scenes(
        #     video_path,
        #     output_json_dir=label_dir,
        #     segments_dir=clip_dir
        # )
    # 각 비디오 처리 파이프라인
    for vid_key, video_path in external_idx_table.items():
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        try:
            stt_result = send_for_stt(video_path)
            stt_path = os.path.join(stt_dir, f"{base_name}.json")
            with open(stt_path, 'w') as f:
                json.dump(stt_result, f)
        except Exception as e:
            print(f"STT 처리 실패: {str(e)}")
    # 3. Summary 생성
    for vid_key, video_path in external_idx_table.items():
        try:
            summary_result = send_for_summary(video_path)
            summary_path = os.path.join(summary_dir, f"{base_name}.json")
            with open(summary_path, 'w') as f:
                json.dump(summary_result, f)
        except Exception as e:
            print(f"요약 생성 실패: {str(e)}")
        gr.Info("✅ summary 생성 완료 (전처리중...)")
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

            gr.Info("✅ caption 생성 완료 (전처리중...)")
            print(f"video_id: {video_id}, base_name: {base_name}, vision_json_path: {vision_json_path}, speech_json_path: {speech_json_path}, summary_dir: {summary_dir}, caption: {caption}")

            # JSON 파일 읽기 및 데이터베이스 저장
            with open(vision_json_path, 'r') as f:
                json_data = json.load(f)
                
            video_id = os.path.splitext(clip_file)[0]
            segment_name = video_id
            start_time = json_data[video_id]['start_time']
            end_time = json_data[video_id]['end_time']
            caption_en = json_data[video_id]['caption']
            caption_ko = translate_deepl(caption_en, 'en', 'ko')

            # Elasticsearch에 저장  
            client = get_elasticsearch_client()
            embedder = VideoCaption()
            embedding = embedder.generate_embedding(caption_en)
            
            embedder.save_to_elasticsearch(
                client=client,
                segment_name=segment_name,
                start_time=start_time,
                end_time=end_time,
                caption=caption_en,
                caption_ko=caption_ko,
                embedding=embedding
            )

    if external_idx_table:
        file_list_str = "\n".join([f"{key}: {os.path.basename(value)}" for key, value in external_idx_table.items()])
        gr.Info("✅ 동영상 제출이 완료되었습니다! (전처리 완료!)")
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
    inputs=[gr.Textbox(label="Video ID",placeholder="예시: video1 or external1"), gr.Textbox(label="Timestamp_start(HH:MM:SS)", placeholder="00:00:00"), 
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
            for i in range(0, 10, 2):  # 2개씩 한 줄에 배치 (총 5줄)
                with gr.Row():
                    video_inputs.append(gr.Video(label=f"동영상 {i+1}"))
                    if i + 1 < 10:  # 10개까지만 추가
                        video_inputs.append(gr.Video(label=f"동영상 {i+2}"))

        with gr.Row():
            with gr.Column():
                clear_btn = gr.Button("삭제", size='lg')
            with gr.Column():
                submit_btn = gr.Button("제출", size='lg')
           
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
                        video_id_input = gr.Textbox(label="Video ID",placeholder="예시: video1 or external1")
                        timestamp_input = gr.Textbox(label="Timestamp(HH:MM:SS)", placeholder="00:00:00")
                        with gr.Row(2):
                            clear_btn = gr.Button("Clear", size='lg')
                            submit_btn = gr.Button("submit", size='lg')
                    with gr.Column() as output_column:
                        image_output = gr.Image(label="Generated Image")
                        caption_output = gr.Textbox(label="Generated Caption")

                        
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
                    
            with gr.Tab("가산점 평가"):
                with gr.Row(2) as columns:
                    with gr.Column() as input_column:
                        video_id_input = gr.Textbox(label="Video ID",placeholder="예시: video1 or external1")
                        start_time_input = gr.Textbox(label="Start time(HH:MM:SS)", placeholder="00:00:00")
                        end_time_input = gr.Textbox(label="End time(HH:MM:SS)", placeholder="00:00:00")
                        with gr.Row(2):
                            clear_btn = gr.Button("Clear", size='lg')
                            submit_btn = gr.Button("submit", size='lg')
                    with gr.Column() as output_column:
                        video_output = gr.Video(label="Generated Clip")
                        caption_output = gr.Textbox(label="Generated Caption")

                        # json_btn = gr.Button("JSON으로 저장", size='lg')
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
                    
    # T2V 탭
    with gr.Tab("Text to Video"):
        with gr.Tabs():
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
    """STT 서버로 비디오 전송"""
    url = ServiceConfig.STT_URL  # 설정 사용
    
    with open(video_path, 'rb') as video_file:
        files = {'video': (os.path.basename(video_path), video_file, 'video/mp4')}
        response = requests.post(url, files=files, timeout=30)
    
    if response.status_code != 200:
        raise Exception(f"STT 서버 오류: {response.text}")
    
    return response.json()

def send_for_summary(video_path: str) -> Dict[str, Any]:
    """요약 서버로 비디오 전송"""
    url = ServiceConfig.SUMMARY_URL  # 설정 사용
    
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
    external_idx_table = {}
    demo.launch(
        debug=True,       # 디버그 모드 활성화 (에러 상세 출력)
        show_error=True,  # UI에 에러 직접 표시
        server_port=30900
    )
