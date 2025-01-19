import os
import json
from typing import Dict, List, Any

# 기존 JSON 파일들이 저장된 폴더 경로와 새로 저장할 JSON 파일 경로
input_folder = r"your path"  # JSON 파일들이 있는 폴더 경로
output_file = "./D3_DR_train_label.json"  # 결과 JSON 파일 경로
genre_key = "DR"

# 전체 annotaion를  JSON 구조로 저장할 딕셔너리
result = {
    "category": genre_key,
    "videos": []
}

# 파일 처리 함수
def process_json_files() -> int:
    """timestamp 범위에 해당하는 비디오의 segment에 대한 caption을 중심으로 JSON 파일들을 통합하여 Annotation 형식을 통일하는 함수.

    입력 폴더의 각 JSON 파일에서 정의된 Annotation 형식에 맞춰 비디오 정보와 세그먼트 정보를 읽어
    통합된 형식의 JSON 파일을 생성한다. 

    Returns:
        int: 처리된 JSON 파일의 개수

    출력 JSON 구조:
    {
        "category": str,
        "videos": [
            {
                "video_id": str,
                "segments": [
                    {
                        "segment_id": str,
                        "start_time": str,
                        "end_time": str,
                        "caption": str,
                        "caption_ko": str
                    },
                    ...
                ]
            },
            ...
        ]
    }
    """
    count = 0 # 변환할 JSON 파일 개수
    
    # 1. 폴더의 json 파일 하나씩 읽기 
    for file_name in sorted(os.listdir(input_folder)):
        if file_name.endswith(".json"):
            file_path = os.path.join(input_folder, file_name)

            # JSON 파일 열기 - utf-8-sig 인코딩 사용
            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            
            #1. JSON 파일 개별별로 개정판 json 에 값 넣기 
            # video_name 읽기
            video_id = data["video_name"].split(".")[0]  # 확장자(.mp4) 제거

            # 개정판 JSON의 비디오 구조 생성
            video_entry = {
                "video_id": video_id,
                "segments": []
            }

            # 각 segement  정보 처리(segment_id , sentence_ko, sentence_en , timestamp)
            for sentence in data.get("sentences", []):
                print(sentence)
                timestamps = sentence["timestamps"] # 리스트
                start_time = format_time(timestamps[0])
                end_time = format_time(timestamps[1])
                
                # 새로운 segment 구조 생성
                segment = {
                    "segment_id": f"{video_id}_{start_time.replace(':', '_')}_{end_time.replace(':', '_')}",
                    "start_time": start_time,
                    "end_time": end_time,
                    "caption": sentence["sentences_en"],
                    "caption_ko": sentence["sentences_ko"]
                }

                video_entry["segments"].append(segment)
            
            # 비디오 1개를 전체 JSON 에 구조에 추가
            result["videos"].append(video_entry)
        
            print("----------------------------------------")
            print(f"{file_name} 내용을 {output_file}에 성공적으로 합쳤습니다. ")
            print(f"저장된 내용 :")
            print(f"{video_entry}")
            count+=1

    # 최종 결과 JSON 파일로 저장
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    return count 

# 시간을 원하는 형식으로 변환하는 함수
def format_time(timestamp: str) -> str:
    """타임스탬프 문자열을 'HH:MM:SS' 형식으로 변환한다

    Args:
        timestamp (str): 변환할 타임스탬프 문자열 (예: "MM:SS.ms" 형식)

    Returns:
        str: 'HH:MM:SS' 형식으로 변환된 시간 문자열

    Example:
        >>> format_time("01:23.456")
        "00:01:23"
    """
    # 소수점 제거 및 00: 추가
    time_parts = timestamp.split(".")[0]  # 소수점 제거
    h, m, s = "00", *time_parts.split(":")
    return f"{h}:{m}:{s}"

# 함수 실행
count = process_json_files()

print(f"=> 변환된 {count} 개 JSON 파일이 {output_file}에 저장되었습니다.")
