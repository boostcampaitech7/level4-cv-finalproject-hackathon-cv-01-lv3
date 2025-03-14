import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model.sources.model_config import VideoChat2Config
from model.sources.modeling_videochat2 import InternVideo2_VideoChat2
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np

class VideoCaption:
    def __init__(self, current_dir):
        # 설정 로드
        self.config = VideoChat2Config.from_json_file(os.path.join(current_dir, "model/configs/config.json"))

        model_path = os.path.join(current_dir, "model/weights")
    
        # 토크나이저 초기화 (Mistral-7B)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_config.llm.pretrained_llm_path,
            trust_remote_code=True,
            use_fast=False,
            token=os.getenv('HF_TOKEN')
        )

        # 모델 초기화
        if torch.cuda.is_available():
            self.model = InternVideo2_VideoChat2.from_pretrained(
                model_path,
                config=self.config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).cuda()
 
        else:
            self.model = InternVideo2_VideoChat2.from_pretrained(
                model_path,
                config=self.config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        self.model.eval()

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

    def generate_caption(self, media_path, media_type='video'):
        """비디오 캡션 생성"""

        # 비디오 로드 및 전처리
        if media_type == 'video':
            video_tensor = self.load_media(
                media_path, 
                media_type=media_type,
                num_segments=self.config.model_config.vision_encoder.num_frames
            )
        
        elif media_type == 'image':
            video_tensor = self.load_media(
                media_path, 
                media_type=media_type,
                num_segments=1
            )
        
        if torch.cuda.is_available():
            video_tensor = video_tensor.cuda()

        # 채팅 히스토리 초기화
        chat_history = []
        
        # 캡션 생성
        response, _ = self.model.chat(
            tokenizer=self.tokenizer,
            msg='',
            user_prompt='Describe the video step by step',
            instruction="Carefully watch the video and describe what is happening in detail.",
            media_type=media_type,
            media_tensor=video_tensor,
            chat_history=chat_history,
            return_history=True,
            generation_config={
                'do_sample': False,
                'max_new_tokens': 512,
            }
        )
        
        return response

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # VideoCaption 인스턴스 생성
    captioner = VideoCaption(current_dir)
    
    # 비디오 경로 설정
    media_path = os.path.join(current_dir, 'data', "본인이 원하는 비디오 경로")
    
    # 캡션 생성
    caption = captioner.generate_caption(media_path, media_type='video')
    print("Generated Caption:", caption)

if __name__ == "__main__":
    main()