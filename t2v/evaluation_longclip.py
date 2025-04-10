import torch
import os
import sys
base_dir = os.path.dirname(os.path.abspath(__file__))  # level4 폴더 기준
weights_path = os.path.join(base_dir, "weights")
sys.path.append(weights_path)  # weights 폴더를 모듈 경로에 추가
from LongCLIP.model import longclip  # 수정된 import 경로
import numpy as np
import cv2
from PIL import Image
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LongCLIP:
    """
    Frame-Text 간 Similarity를 비교할 수 있는 LongCLIP 모델을 가용할 수 있는 클래스
    """
    def __init__(self):
        # LongCLIP 모델 로드
        base_dir = os.path.dirname(os.path.abspath(__file__))  # level4 폴더 기준
        weights_path = os.path.join(base_dir, "weights", "LongCLIP","checkpoints","longclip-L.pt")
        self.model, self.preprocess = longclip.load(weights_path, DEVICE)
    
    def _sample_video_frames(self, video_path: str, num_frames: int=8, target_size: tuple=(224,224)): 
        """Video를 받아 num_frames 수만큼 Sampling을 하여 반환해주는 함수"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 균등 간격으로 n개 프레임 선택 
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        frames = []

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                frame = cv2.resize(frame, target_size)  # 프레임 크기 조정
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB 변환
                frames.append(frame)

        cap.release()
        print(f"total frames: {total_frames}")
        
        return np.array(frames)  # (num_frames, 224, 224, 3) 형태 반환

    def encode_video(self, video_path: str, num_frames: int=8):
        """비디오를 받아 프레임 단위로 임베딩하는 함수"""
        frames = self._sample_video_frames(video_path, num_frames)  # (num_frames, 224, 224, 3)

        # 프레임을 PIL 이미지로 변환한 후 preprocess 적용
        frames_pil = [Image.fromarray(frame) for frame in frames]  # numpy.ndarray -> PIL.Image
        frames_tensor = torch.stack([self.preprocess(frame) for frame in frames_pil]).to(DEVICE)  # (num_frames, 3, 224, 224)
        
        # 각 프레임을 개별적으로 인코딩하고 평균내기
        with torch.no_grad():
            frame_features = [self.model.encode_image(frame.unsqueeze(0)) for frame in frames_tensor]
        
        # 각 프레임의 임베딩을 정규화
        frame_embeddings = [{"frame_index": i, 
                            "frame_embedding": (frame_feature.squeeze() / frame_feature.norm(dim=-1, keepdim=True))}
                            for i, frame_feature in enumerate(frame_features)]
        
        return frame_embeddings

    def encode_text(self, text: str):
        """텍스트를 받아 임베딩하는 함수"""
        text_input = longclip.tokenize([text]).to(DEVICE)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        return text_features / text_features.norm(dim=-1, keepdim=True)