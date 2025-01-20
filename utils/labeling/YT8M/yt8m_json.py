import os
import json
from typing import Dict, List, Any
import pandas as pd

# 기존 JSON 파일들이 저장된 폴더 경로와 새로 저장할 JSON 파일 경로
csv_file_path = "./yt8m_seg.csv"  # 통합 annotation의 CSV 경로
output_dir =  "./labels" #결과 JSON 파일들을  저장할 디렉토리
# genre_key = "MovieClip"

# JSON 파일로 저장하는 함수
def save_segment_as_json(row):
    """
    YT8M 데이터셋의 통합 CSV 속 각 행(Segment)정보 마다 개별적인 JSON 파일들로 저장
    
    Args : 
        row (pd.Series): 세그먼트 정보가 담긴 pandas Series 객체 
        
        Required:
                - segment_id (str): 세그먼트 식별자
                - start_time (int): 시작 시간 (초 단위)
                - end_time (int): 종료 시간 (초 단위)
                - caption (str): segment 캡션 (없으면 NAN으로 저장됨)
     Returns:
        None : JSON 파일로 저장됨

    """
    segment_id = row['segment_id']
    start_time = row['start_time']
    end_time = row['end_time']
    caption = row['caption'] if pd.notna(row['caption']) else "nan"
    
    # 시간을 'HH:MM:SS' 형식으로 변환
    start_time_str = f"{start_time // 60:02}:{start_time % 60:02}:00"
    end_time_str = f"{end_time // 60:02}:{end_time % 60:02}:00"
    
    # JSON 파일 내용
    segment_data = {
        segment_id: {
            "start_time": start_time_str,
            "end_time": end_time_str,
            "caption": caption
        }
    }
    # 파일명
    file_name = f"{segment_id}.json"
    file_path =  os.path.join(output_dir, file_name)
    # JSON 파일로 저장
    with open(file_path, 'w') as json_file:
        json.dump(segment_data, json_file, indent=2)

# 디렉토리가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# DataFrame으로 변환
df = pd.read_csv(csv_file_path)

# 각 row별로 JSON 파일 저장
for _, row in df.iterrows():
    save_segment_as_json(row)