import tensorflow as tf
import os
from pathlib import Path
from typing import List, Set

def combine_txt_files(input_dir: str, output_path: str) -> None:
    """
    주어진 디렉토리의 모든 txt 파일들을 읽어서 중복을 제거하고 하나의 파일로 저장
    
    Args:
        input_dir (str): txt 파일들이 있는 디렉토리 경로
        output_path (str): 결과를 저장할 파일 경로
    """
    # 입력 디렉토리의 모든 txt 파일 찾기
    txt_files: List[str] = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    # 모든 ID를 저장할 set 생성 (중복 제거를 위해 set 사용)
    all_ids: Set[str] = set()
    
    # 각 txt 파일을 읽어서 ID들을 set에 추가
    for txt_file in txt_files:
        file_path = os.path.join(input_dir, txt_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            ids = f.read().splitlines()  # 각 줄을 리스트로 읽기
            all_ids.update(ids)  # set에 추가 (자동으로 중복 제거)
    
    # 정렬된 ID들을 새로운 파일에 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 출력 디렉토리 생성
    with open(output_path, 'w', encoding='utf-8') as f:
        for id_value in sorted(all_ids):
            f.write(id_value + '\n')
    
    print(f'총 {len(all_ids)}개의 고유 ID가 {output_path} 파일에 저장되었습니다.')

# ... existing code ...

# 파일 저장 후 중복 제거 및 합치기
if __name__ == "__main__":
    # 프로젝트 루트에서 실행된다고 가정하고 경로 설정
    input_directory = "./data/yt8m/frame"  # tfrecord가 저장된 디렉토리
    output_file = "./data/yt8m/combined_found_videos.txt"  # 결과 파일 저장 경로
    
    # 디렉토리가 없으면 생성
    os.makedirs(input_directory, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    combine_txt_files(input_directory, output_file)