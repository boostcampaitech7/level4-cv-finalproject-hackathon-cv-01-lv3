import yt_dlp
import ffmpeg  # Make sure this is `ffmpeg-python`
import os
import shutil
from torch.utils.data import DataLoader
from decodeTFrecord import YoutubeSegmentDataset, collate_segments

def download_video(url, filename):
    """
    유튜브 비디오를 다운로드하는 함수입니다.
    url: 다운로드하고자 하는 유튜브 비디오의 url
    filename: 저장할 이름
    """
    ydl_opts = {
        'ffmpeg_location': '/opt/conda/bin/ffmpeg',  # ffmpeg 설치 경로
        'format': 'mp4',
        'outtmpl': filename,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # 파일 이동
    target_path = os.path.join("./origin", filename)
    shutil.move(filename, target_path)
    
    return target_path

def cut_video_ffmpeg(input_file, output_file, start_time, end_time):
    """
    FFmpeg를 사용하여 비디오를 자르는 함수입니다.
    input_file: 원본 비디오 파일 경로
    output_file: 자를 비디오를 저장할 파일 경로
    start_time: 자를 시작 시간 ('시:분:초' 또는 초 단위)
    end_time: 자를 종료 시간 ('시:분:초' 또는 초 단위)
    """
    ffmpeg.input(input_file, ss=start_time, to=end_time).output(output_file).run()

def sec_to_time(sec):
    """
    초(sec)을 시:분:초 형식으로 변환하는 함수입니다.
    sec: 변환할 초 단위 시간
    """
    s = sec % 60
    m = sec // 60
    h = sec // 3600
    return f"{h:02d}:{m:02d}:{s:02d}"

folder_name = 'clips'

videos = [
    "https://www.youtube.com/watch?v=KYxUZoMU9Nc",
    # "https://www.youtube.com/watch?v=0Dy2fo6E_pI",
    # "https://www.youtube.com/watch?v=AWJPmyL5uHI",
    # "https://www.youtube.com/watch?v=JU189rHBIIQ"
]

# seg 예시
file_path = "./*.tfrecord"
typ = 'seg'
dataset = YoutubeSegmentDataset(typ, file_paths=file_path)
loader = DataLoader(dataset, num_workers=0, batch_size=5, collate_fn=collate_segments)

os.makedirs('./origin', exist_ok=True)

for (vids, video_datas, video_masks, segments, segment_start_times, labels, named_labels) in loader:
    for (vid, video_data, video_mask, segment, segment_start_time, label, named_label) in zip(videos, video_datas, video_masks, segments, segment_start_times, labels, named_labels):
    
        # 원본 비디오가 없으면 다운로드
        if not os.path.exists(f"./origin/{vid[-11:]}.mp4"):
            origin_path = download_video(vid, f"{vid[-11:]}.mp4")
        
        # start_time, end_time 설정
        start_time = sec_to_time(int(segment_start_time))
        end_time = sec_to_time(int(segment_start_time) + 5)
        
        # 각 비디오에 대해 폴더 생성
        new_dir = f"./{folder_name}/{vid[-11:]}"
        os.makedirs(new_dir, exist_ok=True)
        
        # FFmpeg로 비디오 자르기 및 저장
        output_path = f"{new_dir}/{vid[-11:]}_{start_time.replace(':', '_')}-{end_time.replace(':', '_')}.mp4"
        cut_video_ffmpeg(f"./origin/{vid[-11:]}.mp4", output_path, start_time, end_time)
