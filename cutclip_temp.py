from moviepy import VideoFileClip
from decodeTFrecord import YoutubeSegmentDataset, collate_segments
from pytubefix import YouTube
from pytubefix.cli import on_progress
from torch.utils.data import DataLoader
import os
def download_video(url,filename):
    """
    유튜브 비디오를 다운로드하는 함수입니다.
    url: 다운로드하고자 하는 유튜브 비디오의 url
    filename: 저장할 이름
    """
    yt = YouTube(url, on_progress_callback = on_progress, use_oauth=True, allow_oauth_cache=True)
    # 다운로드 실행 
    ys = yt.streams.get_highest_resolution()
    ys.download(filename=filename)
    
    # 파일 이동
    target_path = os.path.join("./origin", filename)
    shutil.move(filename, target_path)
    
    return target_path
    
def make_clip_video(path, save_path, start_t, end_t):
    """
    input 비디오를 자를 수 있는 함수입니다.
    path: 자를 대상이 되는 비디오(input 비디오)의 경로
    save_path: 비디오를 자른 clip이 저장될 경로
    start_t: timestamp의 시작점 (형식: "시:분:초" ex: "00:00:04")
    end_t: timestamp의 끝점 (형식: "시:분:초" ex: "00:00:04") 
    """
    clip_video = VideoFileClip(path).subclipped(start_t, end_t)
    clip_video.write_videofile(save_path, codec="libx264", audio_codec="aac")

def sec_to_time(sec):
    """
    초(sec)을 시:분:초로 변환할 수 있는 함수입니다.
    sec: 특정 시점의 초(sec)
    """
    s = sec % 60
    m = sec // 60
    h = sec // 3600
    return f"{h:02d}:{m:02d}:{s:02d}"

folder_name = 'clips'

videos = ["https://www.youtube.com/watch?v=KYxUZoMU9Nc",
        #   "https://www.youtube.com/watch?v=0Dy2fo6E_pI",
        #   "https://www.youtube.com/watch?v=AWJPmyL5uHI",
        #   "https://www.youtube.com/watch?v=AWJPmyL5uHI",
        #   "https://www.youtube.com/watch?v=JU189rHBIIQ"
]

# seg 예시
file_path = "./*.tfrecord"
typ = 'seg'
dataset = YoutubeSegmentDataset(typ, file_paths=file_path)
loader = DataLoader(dataset, num_workers=0, batch_size=5, collate_fn = collate_segments)

os.makedirs('./origin', exist_ok=True)

for (vids, video_datas, video_masks, segments, segment_start_times, labels, named_labels) in loader:
    for (vid, video_data, video_mask, segment, segment_start_time, label, named_label) in zip(videos, video_datas, video_masks, segments, segment_start_times, labels, named_labels):
    
        # # 해당 원본 비디오가 없으면 다운로드 (download_video 함수가 제대로 작동할 시 활성화)
        # if not os.path.exists(f"./origin/{vid[-11:]}.mp4"):
        #     origin_path = download_video(vid, f"{vid[-11:]}.mp4")
        
        # start_time, end_time 설정 
        start_time = sec_to_time(int(segment_start_time))
        end_time = sec_to_time(int(segment_start_time)+5)
        
        # Video 별로 폴더 따로 만들기
        new_dir = f"./{folder_name}/{vid[-11:]}"
        os.makedirs(new_dir, exist_ok=True)
        
        # Video 클리핑 이후 저장
        make_clip_video(f"./origin/{vid[-11:]}.mp4", f"{new_dir}/{vid[-11:]}_{start_time.replace(':','_')}-{end_time.replace(':','_')}.mp4", start_time, end_time)