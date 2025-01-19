import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model_config import VideoChat2Config
from modeling_videochat2 import InternVideo2_VideoChat2
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T

class VideoCaption:
    def __init__(self, model_path):
        # 설정 로드
        self.config = VideoChat2Config.from_json_file(os.path.join('config.json'))
        
        # 토크나이저 초기화 (Mistral-7B)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_config.llm.pretrained_llm_path,
            trust_remote_code=True,
            use_fast=False,
            token=os.getenv('HF_TOKEN')
        )

        # 모델 초기화
        if torch.cuda  .is_available():
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

    def load_video(self, video_path, num_segments=8, resolution=224, hd_num=6):
        """비디오 로드 및 전처리"""
        vr = VideoReader(video_path, ctx=cpu(0))
        
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

    def generate_caption(self, video_path):
        """비디오 캡션 생성"""
        # 비디오 로드 및 전처리
        video_tensor = self.load_video(
            video_path, 
            num_segments=self.config.model_config.vision_encoder.num_frames
        )
        
        if torch.cuda.is_available():
            video_tensor = video_tensor.cuda()

        # 채팅 히스토리 초기화
        chat_history = []
        
        # 캡션 생성
        response, _ = self.model.chat(
            tokenizer=self.tokenizer,
            msg='The video starts with a man and a dog playing in the snow. The man is kneeling down and the dog is running around him. The man is wearing a black jacket and blue jeans. The dog is a light brown color and has a black nose. The snow is falling down from the sky and covering the ground. The man and the dog are having fun playing in the snow. The man is throwing snow at the dog and the dog is running around and playing with the snow. The man is also throwing snow at the camera and the dog is running around and playing with the snow. The man and the dog are having a great time playing in the snow. The video ends with the man and the dog still playing in the snow.',
            user_prompt='Describe the video step by step',
            instruction="Carefully watch the video and describe what is happening in detail.",
            media_type='video',
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
    # 모델 경로 설정
    model_path = os.path.join(current_dir)
    
    # VideoCaption 인스턴스 생성
    captioner = VideoCaption(model_path)
    
    # 비디오 경로 설정
    video_path = os.path.join(current_dir.split("tasks/captioning/InternVideo2-Chat-8B")[0], "demo/example1.mp4")
    
    # 캡션 생성
    caption = captioner.generate_caption(video_path)
    print("Generated Caption:", caption)

if __name__ == "__main__":
    main()