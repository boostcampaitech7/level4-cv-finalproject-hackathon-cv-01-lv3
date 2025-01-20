from moviepy import VideoFileClip
from decodeTFrecord import YoutubeSegmentDataset, collate_segments
import yt_dlp
from torch.utils.data import DataLoader
import os
import shutil
def download_video(url, filename):
    """
    유튜브 비디오를 다운로드하는 함수입니다.
    url: 다운로드하고자 하는 유튜브 비디오의 url
    filename: 저장할 이름
    """
    ydl_opts = {
        'ffmpeg_location': 'which ffmpeg 명령어로 경로 찾아 넣어주세요',  #### 바꿔주세요 ####
        'format': 'mp4',
        'outtmpl': filename,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
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
def vid_to_url_dict(file_path):
    """
    vid를 youtube id로 바꾸기 위해 {vid:youtube id}쌍의 딕셔너리를 생성합니다.
    """
    result = {}
    with open(file_path, 'r') as f:
        content = f.readlines()  
    for line in content:
        line = line.strip()
        if line:
            try:
                vid, url = line.split()
                result[vid] = url
            except ValueError:
                print(f"Invalid line: {line}")
    return result
def make_video_list(loader, vid_to_url):
    """
    vid와 매칭되는 youtube id를 찾아 url의 형태로 이를 리스트에 저장합니다.
    """
    result = []
    for vids, _, _, _, _, _, _ in loader:
        for vid in vids:
            url = vid_to_url.get(vid)
            if url:
                result.append(f"https://www.youtube.com/watch?v={url}")
            else:
                print(f"URL not found for vid: {vid}")  # If URL not found, print message
    return result
def get_video_duration(video_path):
    """
    영상 총 길이를 return 합니다.
    """
    with VideoFileClip(video_path) as video:
        return video.duration 
youtube_id_txt = "./data/youtube_id_movieclips.txt"
vid_to_url = vid_to_url_dict(youtube_id_txt)
folder_name = 'clips'
# seg 예시
file_path = "./*.tfrecord"
typ = 'seg'
dataset = YoutubeSegmentDataset(typ, file_paths=file_path)
loader = DataLoader(dataset, num_workers=0, batch_size=5, collate_fn=collate_segments)
videos = make_video_list(loader, vid_to_url)
os.makedirs('./origin', exist_ok=True)
# loader에서 받은 데이터를 활용하여 비디오 처리
for vids, video_datas, video_masks, segments, segment_start_times, labels, named_labels in loader:
    for vid, video_data, video_mask, segment, segment_start_time, label, named_label in zip(videos, video_datas, video_masks, segments, segment_start_times, labels, named_labels):
        # 해당 원본 비디오가 없으면 다운로드 (download_video 함수가 제대로 작동할 시 활성화)
        if not os.path.exists(f"./origin/{vid[-11:]}.mp4"):
            origin_path = download_video(vid, f"{vid[-11:]}.mp4")
        # start_time, end_time 설정 
        video_path = f"./origin/{vid[-11:]}.mp4"
        video_duration = get_video_duration(video_path)
        segment_start_time = int(segment_start_time)
        segment_end_time = segment_start_time + 5
        start_time = sec_to_time(segment_start_time)
        end_time = sec_to_time(segment_end_time)
        if segment_start_time > video_duration:
            break
        if segment_end_time > video_duration:
            end_time = sec_to_time(int(video_duration))
        # Video 별로 폴더 따로 만들기
        new_dir = f"./{folder_name}/{vid[-11:]}"
        os.makedirs(new_dir, exist_ok=True)
        # Video 클리핑 이후 저장
        make_clip_video(f"./origin/{vid[-11:]}.mp4", f"{new_dir}/{vid[-11:]}_{start_time.replace(':','_')}-{end_time.replace(':','_')}.mp4", start_time, end_time)