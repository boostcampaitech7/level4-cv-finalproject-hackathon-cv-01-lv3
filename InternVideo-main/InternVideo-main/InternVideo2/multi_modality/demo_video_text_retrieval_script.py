# 25.01.14, 주피터 노트북 파일 오류로 인해 스크립트 파일 변환, deamin
import sys
import os
import numpy as np
import io
import cv2
import torch

from demo_config import Config, eval_dict_leaf
from demo.utils import retrieve_text, _frame_from_video, setup_internvideo2

def main():
    # 프로젝트 루트 디렉토리를 설정하고 sys.path에 추가
    project_root = os.path.abspath(os.path.join(os.getcwd()))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    print(sys.path)
    
    # 비디오 파일 열기 및 프레임 추출
    video_path = os.path.join('demo', 'example1.mp4')
    video = cv2.VideoCapture(video_path)
    frames = [frame for frame in _frame_from_video(video)]
    
    # 텍스트 후보군 정의
    text_candidates = [
        "A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon.",
        "A man in a gray coat walks through the snowy landscape, pulling a sleigh loaded with toys.",
        "A person dressed in a blue jacket shovels the snow-covered pavement outside their house.",
        "A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner.",
        "A person stands on the snowy floor, pushing a sled loaded with blankets, preparing for a fun-filled ride.",
        "A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees.",
        "A playful dog slides down a snowy hill, wagging its tail with delight.",
        "A person in a blue jacket walks their pet on a leash, enjoying a peaceful winter walk among the trees.",
        "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.",
        "A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery."
    ]
    
    # 설정 파일 로드 및 평가
    config_path = os.path.join('demo', 'internvideo2_stage2_config.py')
    config = Config.from_file(config_path)
    config = eval_dict_leaf(config)
    
    # 모델 및 토크나이저 설정
    intern_model, tokenizer = setup_internvideo2(config)
    
    # 텍스트 검색 수행
    texts, probs = retrieve_text(frames, text_candidates, model=intern_model, topk=5, config=config)
    
    # 결과 출력
    for t, p in zip(texts, probs):
        print(f'text: {t} ~ prob: {p:.4f}')

if __name__ == "__main__":
    main() 