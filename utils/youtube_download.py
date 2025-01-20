import os
import pandas as pd
from load_video import make_video_list, download_video, get_video_duration

### 설정해주세요
youtube_id_txt = "./data/additional_id_movieclips.txt" # {vid youtube_id} 형태로 저장되어 있는 txt파일의 경로
logsheet_path = "./youtube_download_log.csv" # (Optional) 만약, 사용 중인 logsheet.csv가 있다면 경로를 설정해주세요. 없다면 빈칸으로 설정해주세요
prefix_path = "./origin" # 원본 비디오가 저장될 경로
video_type = "yt8m" # 원본 비디오의 출처. {Youtube-8M Dataset: yt8m, AIHUB-비디오장면설명문 데이터: D3, ...(추가해주세요)}
category = "Movieclips" # 원본 비디오의 카테고리
output_csv = "youtube_download_log" # 최종 logsheet이 저장될 이름
###

# youtube_id_txt로부터 다운로드 받아야 할 youtube_id를 받아옴
video_lists = make_video_list(youtube_id_txt)

# 만약, logsheet_path가 빈칸이라면 빈 DataFrame에서부터 시작하고, 사용 중인게 있다면 이어서 작성함
if logsheet_path == "":
    logsheet = pd.DataFrame(columns=['youtube_id', 'status'])
else:
    logsheet = pd.read_csv(logsheet_path)

# 원본 비디오가 저장될 곳 설정
os.makedirs(prefix_path, exist_ok=True)

# 전체에 리스트에 대해서 다운로드
for i, vid_url in enumerate(video_lists):
    video_path = f"./{prefix_path}/{video_type}_{category}_{vid_url[-11:]}.mp4"
    if not os.path.exists(video_path):

        # 비디오 다운로드 시도: 만약 성공한다면 logsheet에 status를 1로 기록, 실패한다면 다운로드 하지 않고 logsheet에 status를 0으로 기록
        try:
            origin_path = download_video(prefix_path, vid_url, f"{video_type}_{category}_{vid_url[-11:]}.mp4")
            video_duration = get_video_duration(video_path)
            new_row = pd.DataFrame([{"youtube_id": vid_url[-11:], "status": 1}])
            logsheet = pd.concat([logsheet, new_row], ignore_index=True)
        except:
            print(f"Failed to download {video_type}_{category}_{vid_url[-11:]}.mp4")
            new_row = pd.DataFrame([{"youtube_id": vid_url[-11:], "status": 0}])
            logsheet = pd.concat([logsheet, new_row], ignore_index=True)

# 최종 logsheet을 csv형태로 변환하여 저장
logsheet.to_csv(f"{output_csv}.csv", index=False)