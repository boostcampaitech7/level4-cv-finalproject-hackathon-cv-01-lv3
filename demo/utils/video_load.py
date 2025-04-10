import os
import shutil
import subprocess
import ffmpeg
import asyncio
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
import cv2
from decord import VideoReader, cpu 
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel   
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from model.sources.model_config import VideoChat2Config
from model.sources.modeling_videochat2 import InternVideo2_VideoChat2

class VideoLoad:
    def __init__(self):
        self.model_path = "OpenGVLab/InternVL2_5-8B-MPO"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
    
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True
        ).eval().to(self.device)

    def load_media(self, media_path, media_type='video', num_segments=8, resolution=224, hd_num=6):
        """비디오 로드 및 전처리"""
        if media_type == 'video':
            vr = VideoReader(media_path, ctx=cpu(0))

            # 균일한 간격으로 프레임 인덱스 추출
            num_frames = len(vr)
            indices = self._get_frame_indices(num_frames, num_segments)
            
            # 프레임 추출 및 전처리
            frames = vr.get_batch(indices).asnumpy()
            # NumPy 배열을 PyTorch Tensor로 변환
            frames = torch.from_numpy(frames)
            frames = frames.permute(0, 3, 1, 2)  # (N, C, H, W)
            
            # 정규화
            frames = self._transform_frames(frames, resolution)
            
            return frames.unsqueeze(0)  # 배치 차원 추가

        elif media_type == 'image':
            image = Image.open(media_path)
            image = image.resize((resolution, resolution))
            image = np.array(image)
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)  # (C, H, W)
            image = image.unsqueeze(0)  # 배치 차원 추가
            return image
    
    def _get_frame_indices(self, num_frames, num_segments):
        """균일한 간격으로 프레임 인덱스 추출"""
        seg_size = float(num_frames - 1) / num_segments
        indices = torch.linspace(0, num_frames-1, num_segments).long()
        return indices

    def _transform_frames(self, frames, resolution):
        """프레임 전처리"""
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        transform = T.Compose([
            T.Resize((resolution, resolution)),
            T.Lambda(lambda x: x.float().div(255.0)),
            T.Normalize(mean, std)
        ])
        
        return transform(frames)

    def preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """InternVL2.5에 맞춘 프레임 전처리"""
        transform = transforms.Compose([
            transforms.Resize((448, 448)),  # 모델 입력 크기 변경
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(frames.float() / 255.0)

    def generate_caption(self, video_tensor, media_type='video', user_prompt=''):
        if torch.cuda.is_available():
            video_tensor = video_tensor.to(self.device, dtype=torch.bfloat16)

        # 모델 입력 형식에 맞게 변환
        pixel_values = self.preprocess_frames(video_tensor)
        num_patches_list = [video_tensor.shape[0]]  # 프레임 수 기반 패치 리스트 생성

        # 모델 추론
        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            question=user_prompt,
            generation_config={'max_new_tokens': 512, 'do_sample': False}
        )
        
        return response
