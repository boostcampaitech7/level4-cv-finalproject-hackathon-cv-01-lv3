from moviepy import VideoFileClip
import yt_dlp
import shutil
import os

def sec_to_time(sec: int) -> str:
    """
    초(sec)을 시:분:초로 변환할 수 있는 함수입니다.
    sec: 특정 시점의 초(sec)
    """
    s = sec % 60
    m = sec // 60
    h = sec // 3600
    return f"{h:02d}:{m:02d}:{s:02d}"

### 비디오를 다운로드 및 저장하는 데 필요한 함수들
def download_video(prefix: str, url: str, filename: str) -> str:
    """
    유튜브 비디오를 다운로드하는 함수입니다.
    prefix: 원본 비디오가 저장될 디렉토리 경로
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
    target_path = os.path.join(prefix, filename)
    shutil.move(filename, target_path)
    return target_path


def make_video_list(video_lists: str) -> list:
    """
    vid와 매칭되는 youtube id를 찾아 url의 형태로 이를 리스트에 저장합니다.
    video_lists: {vid youtube_id} 형태로 저장되어 있는 txt파일의 경로
    """
    result = []
    with open(video_lists, 'r') as f:
        content = f.readlines()
    for line in content:
        line = line.strip()
        if line:
            try:
                _, youtube_id = line.split()
                result.append(f"https://www.youtube.com/watch?v={youtube_id}")
            except ValueError:
                print(f"Invalid line: {line}")
    return result


def get_video_duration(video_path: str) -> float:
    """
    영상 총 길이를 return 합니다.
    video_path: 영상 총 길이를 보고자 하는 비디오의 경로
    """
    with VideoFileClip(video_path) as video:
        return video.duration


### 비디오를 클립 단위로 자르는 데 필요한 함수
def make_clip_video(path: str, save_path: str, start_t: str, end_t: str) -> None:
    """
    input 비디오를 자를 수 있는 함수입니다.
    path: 자를 대상이 되는 비디오(input 비디오)의 경로
    save_path: 비디오를 자른 clip이 저장될 경로
    start_t: timestamp의 시작점 (형식: "시:분:초" ex: "00:00:04")
    end_t: timestamp의 끝점 (형식: "시:분:초" ex: "00:00:04") 
    """
    clip_video = VideoFileClip(path).subclipped(start_t, end_t)
    clip_video.write_videofile(save_path, codec="libx264", audio_codec="aac", logger=None, threads=8)
    clip_video.close()