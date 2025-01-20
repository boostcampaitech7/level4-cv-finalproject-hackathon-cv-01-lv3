from tqdm import tqdm
import os
import pandas as pd
from load_video import make_clip_video
    
### 설정해주세요
metadata = f"./segments.csv" # CSV형태의 Metadata가 있는 경로
input_video = f"./origin" # mp4 형태의 Video가 있는 경로
output_folder = f"./clips/yt8m" # Cut된 Clip이 저장될 경로
###

def main() -> None:
    meta = pd.read_csv(metadata)

    # 처리할 파일이 없으면 아무것도 하지 않고 바로 반환
    if len(meta) == 0:
        print("No files to be proceeded")
        return

    # CSV 파일 차례대로 처리
    with tqdm(total=len(meta), desc="Processing CSV files", leave=False) as iter_progress:
        for i in range(len(meta)):
            vid = meta.loc[i, "video_id"]
            vtype, category, name = vid.split("_")
            sub_dir = f"{output_folder}/{vtype}/{category}"
            os.makedirs(sub_dir, exist_ok=True)

            segment_id = meta.loc[i, "segment_id"]
            start_time = meta.loc[i, "start_time"]
            end_time = meta.loc[i, "end_time"]
            
            origin_video_path = ".".join([os.path.join(input_video, vid), "mp4"])
            if os.path.exists(origin_video_path):
                output_file_path = os.path.join(sub_dir, f"{segment_id}.mp4")
                make_clip_video(origin_video_path, output_file_path, start_time, end_time)
            else:
                raise ValueError(f"Couldn't find the origin video! {origin_video_path}")
        

            # CSV 파일 진행 상황 업데이트
            iter_progress.update(1)

if __name__ == "__main__":
    main()