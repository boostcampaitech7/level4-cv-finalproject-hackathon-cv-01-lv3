import sys
import os
import numpy as np
import io
import cv2
import torch

from demo_config import Config, eval_dict_leaf
from utils import retrieve_text, _frame_from_video, setup_internvideo2, extract_vid_embedding, extract_txt_embedding, retrieve_video

def main():
    # 프로젝트 루트 디렉토리를 설정하고 sys.path에 추가
    project_root = os.path.abspath(os.path.join(os.getcwd()))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    print(sys.path)
    
    # 비디오 파일 열기 및 프레임 추출
    # video_path = os.path.join('demo', 'example1.mp4')
    video_path = '/data/ephemeral/home/hanseonglee_t2v/level4-cv-finalproject-hackathon-cv-01-lv3/t2v/example1.mp4' # 예시 비디오 경로
    video = cv2.VideoCapture(video_path)
    frames = [frame for frame in _frame_from_video(video)]
    
    # 텍스트 후보군 정의
    # text_candidates = [
    #     "A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon.",
    #     "A man in a gray coat walks through the snowy landscape, pulling a sleigh loaded with toys.",
    #     "A person dressed in a blue jacket shovels the snow-covered pavement outside their house.",
    #     "A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner.",
    #     "A person stands on the snowy floor, pushing a sled loaded with blankets, preparing for a fun-filled ride.",
    #     "A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees.",
    #     "A playful dog slides down a snowy hill, wagging its tail with delight.",
    #     "A person in a blue jacket walks their pet on a leash, enjoying a peaceful winter walk among the trees.",
    #     "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.",
    #     "A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery."
    # ]
    text_candidates = ["there is snow","what the hell is this","sunny day", 'a woman is in the video', 'chatgpt is good for everything', 'A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery']
    # 설정 파일 로드 및 평가
    # config_path = os.path.join('demo', 'internvideo2_stage2_config.py')
    config_path = '/data/ephemeral/home/hanseonglee_t2v/level4-cv-finalproject-hackathon-cv-01-lv3/t2v/models/internvideo2_stage2_config.py'
    config = Config.from_file(config_path)
    config = eval_dict_leaf(config)
    
    # 모델 및 토크나이저 설정
    intern_model, tokenizer = setup_internvideo2(config)

    video_paths = ['/data/ephemeral/home/hanseonglee_t2v/level4-cv-finalproject-hackathon-cv-01-lv3/t2v/example4.mp4',
                   '/data/ephemeral/home/hanseonglee_t2v/level4-cv-finalproject-hackathon-cv-01-lv3/t2v/example2.mp4',
                   '/data/ephemeral/home/hanseonglee_t2v/level4-cv-finalproject-hackathon-cv-01-lv3/t2v/example3.mp4',]
    video_embs = []

    top_videos, top_probs = retrieve_video(video_paths, "black something is comming ", intern_model, device=next(intern_model.parameters()).device )
    print(top_videos)
    print(top_probs)
    # for video_path in video_paths:
    #     video = cv2.VideoCapture(video_path)
    #     frames = [frame for frame in _frame_from_video(video)]
    #     # 1. 비디오 임베딩 추출 후 출력
    #     video_emb = extract_vid_embedding(
    #         frames=frames,
    #         model=intern_model,
    #         config=config.model.vision_encoder,
    #         device=next(intern_model.parameters()).device
    #     )
    #     video_embs.append(video_emb)
    


    # 임베딩 정규화 검증
    # emb = extract_vid_embedding(
    #     frames=frames,
    #     model=intern_model,
    #     config=config.model.vision_encoder,
    #     device=next(intern_model.parameters()).device
    # )
    # norm = np.linalg.norm(emb)
    # print(f"Embedding Norm: {norm:.4f}")  # 1.0000 ±0.001 이어야 함

    # # 차원 검증
    # print("Vision Proj Layer:", intern_model.vision_proj)  # Linear(in_features=1408, out_features=512)
    
    # # 비전 임베딩 출력 (실제 연구개발시에는 주석 처리)
    # print("\n" + "="*50)
    # print("Video Embedding Summary:")
    # print(f"Shape: {video_emb.shape}")
    # print(f"Mean: {np.mean(video_emb):.4f}")
    # print(f"Std: {np.std(video_emb):.4f}")
    # print("First 5 elements:", video_emb[0][:5])  # 첫 번째 프레임의 첫 5개 값
    # print("="*50 + "\n")
    

    # # 2. 텍스트 임베딩 출력
    # text_emb = extract_txt_embedding(
    #     text="there is snow", 
    #     model=intern_model,
    #     tokenizer=tokenizer,
    #     device=next(intern_model.parameters()).device
    # )
    
    # print("\n" + "="*50)
    # print("Text Embedding Summary:")
    # print(f"Shape: {text_emb.shape}")
    # print(f"Mean: {np.mean(text_emb):.4f}")
    # print(f"Std: {np.std(text_emb):.4f}")
    # print("First 5 elements:", text_emb[0][:5])  # 첫 번째 토큰의 첫 5개 값
    # print("="*50 + "\n")

    # # 3. 비디오 임베딩과 텍스트 임베딩 유사도 계산
    # similarities = []
    # for video_emb in video_embs:
    #     similarity = float(video_emb @ text_emb.T)
    #     similarities.append(similarity)
    #     print(f"Similarity: {similarity:.4f}")
    # # 최고 유사도 비디오 출력
    # max_similarity = max(similarities)
    # max_index = similarities.index(max_similarity)
    # print(f"Max Similarity: {max_similarity:.4f} at video index {max_index}")

    # 텍스트 검색 수행
    # texts, probs = retrieve_text(frames, text_candidates, model=intern_model, topk=5, config=config)
    
    # 결과 출력
    # for t, p in zip(texts, probs):
        # print(f'text: {t} ~ prob: {p:.4f}')

if __name__ == "__main__":
    main()