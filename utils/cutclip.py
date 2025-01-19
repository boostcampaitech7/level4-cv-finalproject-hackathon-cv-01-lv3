from moviepy import VideoFileClip
import json
from tqdm import tqdm
import os
    
input_label = f"./test" # JSON형태의 Annotation이 있는 경로
input_video = f"D3/Input_video" # mp4 형태의 Video가 있는 경로
output_folder = f"clips" # Cut된 Clip이 저장될 경로

def make_clip_video(path: str, save_path: str, start_t: str, end_t: str) -> None:
    """
    input 비디오를 자를 수 있는 함수입니다.
    path: 자를 대상이 되는 비디오(input 비디오)의 경로
    save_path: 비디오를 자른 clip이 저장될 경로
    start_t: timestamp의 시작점 (형식: "시:분:초" ex: "00:00:04")
    end_t: timestamp의 끝점 (형식: "시:분:초" ex: "00:00:04") 
    """
    clip_video = VideoFileClip(path).subclipped(start_t, end_t)
    clip_video.write_videofile(save_path, codec="libx264", audio_codec="aac", logger=None, threads=4)

def sec_to_time(sec: int) -> str:
    """
    초(sec)을 시:분:초로 변환할 수 있는 함수입니다.
    sec: 특정 시점의 초(sec)
    """
    s = sec % 60
    m = sec // 60
    h = sec // 3600
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    file_names = sorted([f for f in os.listdir(input_label) if f.endswith(".json")])

    # 처리할 파일이 없으면 아무것도 하지 않고 바로 반환
    if len(file_names) == 0:
        print("No files to be proceeded")
        return

    # JSON 파일 차례대로 처리
    with tqdm(total=len(file_names), desc="Processing JSON files", leave=False) as json_progress:
        for file_name in file_names:
            file_path = os.path.join(input_label, file_name)

            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)

            category = data['category']
            sub_dir = f"{output_folder}/{category}"
            os.makedirs(sub_dir, exist_ok=True)

            with tqdm(total=len(data['videos']), desc=f"Processing videos in {file_name}", leave=False) as video_progress:
                for video in data['videos']:
                    vid = video['video_id'] # Video ID

                    # segment_id가 클립의 이름이 됨
                    with tqdm(total=len(video['segments']), desc=f"Processing segments in {vid}", leave=False) as segment_progress:
                        for segments in video['segments']:
                            segment_id = segments['segment_id']
                            start_time = segments['start_time']
                            end_time = segments['end_time']
                            
                            origin_video_path = ".".join([os.path.join(input_video, vid), "mp4"])
                            if os.path.exists(origin_video_path):
                                output_file_path = os.path.join(sub_dir, f"{segment_id}.mp4")
                                make_clip_video(origin_video_path, output_file_path, start_time, end_time)
                            else:
                                raise ValueError(f"Couldn't find the origin video! {origin_video_path}")
                    
                            # 세그먼트 진행 상황 업데이트
                            segment_progress.update(1)
                    
                    # 비디오 진행 상황 업데이트
                    video_progress.update(1)

            # JSON 파일 진행 상황 업데이트
            json_progress.update(1)

if __name__ == "__main__":
    main()