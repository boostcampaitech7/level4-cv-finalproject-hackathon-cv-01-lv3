import os
import json
import csv
from typing import List, Dict, Any

# 기존 라벨 JSON 저장 폴더와 전체 JSON 파일 내 Metadata 저장할 CSV 파일 경로 설정
input_folder = r"yours JSON directory path"  # JSON 파일들이 있는 폴더 경로
output_csv = "./D3_DR_all_meta.csv"  # 저장할 CSV 파일 경로


# 폴더 내 모든 JSON 파일을 처리하여 CSV로 저장하는 함수
def json_to_csv_first_format(input_folder: str, output_csv: str) -> None:
    """JSON 파일들의 메타데이터를 CSV 파일로 변환하는 함수입니다.

    폴더 input_folder 에서 모든 JSON 파일을 읽어 비디오 메타데이터와 문장 정보를 CSV 형식으로 변환한다. 
    각 문장에는 0부터 시작하는 고유한 sentence_id가 부여된다. 

    Args:
        input_folder (str): JSON 파일들이 저장된 폴더의 경로
        output_csv (str): 저장될 CSV 파일의 경로

    Returns:
        None

    CSV 컬럼 구조:
        - video_name: 비디오 파일명
        - width: 비디오 너비
        - height: 비디오 높이
        - frame_rate: 프레임 레이트
        - duration: 비디오 길이
        - total_frame: 총 프레임 수
        - film_method: 촬영 방법
        - filmed_date: 촬영 날짜
        - domain_id: 도메인 ID
        - place: 촬영 장소
        - f1_consis: F1 일관성 점수
        - f1_consis_avg: F1 일관성 평균
        - annotated_date: 주석 작성 날짜
        - version: 버전
        - revision_history: 수정 이력
        - seg_annotator_id: 세그먼트 주석자 ID
        - seg_confirmer_id: 세그먼트 확인자 ID
        - distributor: 배포자
        - describe_ko: 비디오의 한국어 caption
        - describe_en: 비디오의 영어 caption
        - timestamp_start: 시작 타임스탬프
        - timestamp_end: 종료 타임스탬프
        - sentence_ko: Sentence(segment)의 한국어 caption
        - sentence_en: Sentence(segment)의  영어 caption
        - sentence_id: sentence ID (0부터 시작)
    """
    # CSV 파일의 헤더 정의
    csv_header = [
        "video_name", "width", "height", "frame_rate", "duration", "total_frame", 
        "film_method", "filmed_date", "domain_id", "place", "f1_consis", 
        "f1_consis_avg", "annotated_date", "version", "revision_history", 
        "seg_annotator_id", "seg_confirmer_id", "distributor", 
        "describe_ko", "describe_en", "timestamp_start", "timestamp_end", 
        "sentence_ko", "sentence_en", "sentence_id"
    ]
    
    file_count = 0  # 처리한 JSON 파일 개수
    
    # CSV 파일 생성
    with open(output_csv, mode='w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)  # 헤더 작성
        
        # 폴더 내 JSON 파일 읽기
        for filename in os.listdir(input_folder):
            if filename.endswith(".json"):  
                file_count += 1  # 처리하는 JSON 파일 개수 
    
                json_path = os.path.join(input_folder, filename)
                
                with open(json_path, mode='r', encoding='utf-8-sig') as jsonfile:
                    data = json.load(jsonfile)
                    
                    # JSON에서 Video의 meta 데이터 추출
                    video_name = data.get("video_name", "")
                    width = data.get("width", "")
                    height = data.get("height", "")
                    frame_rate = data.get("frame_rate", "")
                    duration = data.get("duration", "")
                    total_frame = data.get("total_frame", "")
                    film_method = data.get("film_method", "")
                    filmed_date = data.get("filmed_date", "")
                    domain_id = data.get("domain_id", "")
                    place = data.get("place", "")
                    f1_consis = ";".join(map(str, data.get("f1_consis", [])))
                    f1_consis_avg = data.get("f1_consis_avg", "")
                    annotated_date = data.get("annotated_date", "")
                    version = data.get("version", "")
                    revision_history = data.get("revision_history", "")
                    seg_annotator_id = ";".join(map(str, data.get("seg_annotator_id", [])))
                    seg_confirmer_id = ";".join(map(str, data.get("seg_confirmer_id", [])))
                    distributor = data.get("distributor", "")
                    describe_ko = data.get("describe_ko", "")
                    describe_en = data.get("describe_en", "")
                    
                    # sentences 항목의 데이터를 하나씩 처리
                    for sentence_idx, sentence in enumerate(data.get("sentences", [])):
                        timestamps = sentence.get("timestamps", ["", ""])
                        timestamp_start = timestamps[0]
                        timestamp_end = timestamps[1] if len(timestamps) > 1 else ""
                        sentence_ko = sentence.get("sentences_ko", "")
                        sentence_en = sentence.get("sentences_en", "")
                        
                        # 한 줄씩 CSV에 기록
                        writer.writerow([
                            video_name, width, height, frame_rate, duration, total_frame, 
                            film_method, filmed_date, domain_id, place, f1_consis, 
                            f1_consis_avg, annotated_date, version, revision_history, 
                            seg_annotator_id, seg_confirmer_id, distributor, 
                            describe_ko, describe_en, timestamp_start, timestamp_end, 
                            sentence_ko, sentence_en, sentence_idx
                        ])
    
    # 처리 결과 출력
    print(f"총 {file_count}개의 JSON 파일을 처리하여 CSV 파일로 저장했습니다: {output_csv}")


# 함수 실행
json_to_csv_first_format(input_folder, output_csv)
