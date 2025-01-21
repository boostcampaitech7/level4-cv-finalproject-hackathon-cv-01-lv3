from tqdm import tqdm
import os
import pandas as pd
from load_video import make_clip_video, make_image_video
import math

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

            segment_id = meta.loc[i, "segment_id"]
            start_time = meta.loc[i, "start_time"]
            end_time = meta.loc[i, "end_time"]
            duration = meta.loc[i, "duration"]
            origin_video_path = ".".join([os.path.join(input_video, vid), "mp4"])

            # 원본 비디오의 존재 여부 확인
            if not os.path.exists(origin_video_path):
                raise ValueError(f"Couldn't find the origin video! {origin_video_path}")
            
            # start_time이 전체 비디오의 duration 범위 내에 있는지 확인
            if start_time > duration:
                raise ValueError(f"Start Time {start_time} is larger than duration {duration}")
            
            # end_time과 start_time이 같지 않고, end_time이 nan이 아닌 경우, 클립을 만든다고 판단
            # 만약, end_time과 start_time이 같거나, end_time없이 start_time만 주어진 경우, 프레임을 만든다고 판단
            if end_time != start_time and not math.isnan(end_time):
                
                # end_time이 start_time보다 뒤에 있는지 확인
                if start_time > end_time:
                    raise ValueError(f"Start Time {start_time} is larger than End Time {end_time}")
                sub_dir = f"{output_folder_video}/{vtype}/{category}"
                os.makedirs(sub_dir, exist_ok=True)    
                output_file_path = os.path.join(sub_dir, f"{segment_id}.mp4")
                make_clip_video(origin_video_path, output_file_path, start_time, end_time)
            else:
                sub_dir = f"{output_folder_img}/{vtype}/{category}"
                os.makedirs(sub_dir, exist_ok=True)
                output_file_path = os.path.join(sub_dir, f"{segment_id}.jpg")
                make_image_video(origin_video_path, output_file_path, start_time)

            # CSV 파일 진행 상황 업데이트
            iter_progress.update(1)

if __name__ == "__main__":

    ### 설정해주세요
    metadata = f"./segments.csv" # CSV형태의 Metadata가 있는 경로
    input_video = f"./origin" # mp4 형태의 Video가 있는 경로
    output_folder_video = f"./clips" # Cut된 Clip이 저장될 경로
    output_folder_img = f"./frames" # Cut된 Image가 저장될 경로
    ###

    main()