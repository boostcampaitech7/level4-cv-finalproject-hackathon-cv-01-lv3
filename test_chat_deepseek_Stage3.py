import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model.sources.model_config import VideoChat2Config, Qwen2_5_Config
from model.sources.modeling_videochat2_deepseek import InternVideo2_VideoChat2
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np

class VideoCaption:
    def __init__(self, current_dir):
        # 설정 로드
        self.config = VideoChat2Config.from_json_file(os.path.join(current_dir, "model/configs/config_deepseek.json"))
        # self.llm_config = Qwen2_5_Config.from_json_file(os.path.join(current_dir, "model_deepseek/configs/config.json"))
        model_path = os.path.join(current_dir, "model/weights")
    
        # 토크나이저 초기화 (Qwen-14B)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_config.llm.pretrained_llm_path,
            trust_remote_code=True,
            use_fast=False,
            token=os.getenv('HF_TOKEN')
        )
        print(self.config)
        print("tokeizer import 완료")
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
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )
        self.model.eval()
        # 가중치 저장 로직 추가
        # 모델 구조 분석용 디버깅 코드
        # self._analyze_model_structure()
        
        # # 개선된 가중치 저장
        # self._save_structured_weights(model_path)

    # def _analyze_model_structure(self):
    #     """모델 파라미터 키 분석을 위한 도우미 함수"""
    #     print("\n=== Model Structure Analysis ===")
    #     for name, param in self.model.named_parameters():
    #         print(f"Parameter key: {name}")
    #     for name, buffer in self.model.named_buffers():
    #         print(f"Buffer key: {name}")
        
    # def _save_model_weights(self, save_dir):
    #     """모델 구성 요소별 가중치 저장"""
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     # Vision Encoder 가중치 저장
    #     torch.save(
    #         {
    #             'vision_encoder': self.model.vision_encoder.state_dict(),
    #             'vision_layernorm': self.model.vision_layernorm.state_dict()
    #         }, 
    #         os.path.join(save_dir, 'vision_encoder.pth')
    #     )
        
    #     # Q-Former 가중치 저장
    #     torch.save(
    #         {
    #             'qformer': self.model.qformer.state_dict(),
    #             'query_tokens': self.model.query_tokens,
    #             'extra_query_tokens': self.model.extra_query_tokens
    #         },
    #         os.path.join(save_dir, 'qformer.pth')
    #     )
        
    #     # 프로젝션 레이어 가중치 저장
    #     torch.save(
    #         {
    #             'project_up': self.model.project_up.state_dict(),
    #             'project_down': self.model.project_down.state_dict()
    #         },
    #         os.path.join(save_dir, 'projection.pth')
    #     )
    # def _save_structured_weights(self, save_dir):
    #     """키 구조 분석을 기반으로 한 계층적 가중치 저장"""
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     # 계층별 파라미터 그룹화
    #     component_weights = {
    #         'vision': {},
    #         'qformer': {},
    #         'projection': {},
    #         'lm': {},  # 기존 LLM 가중치 (참고용 보관)
    #         'misc': {}
    #     }

    #     # 개선된 파라미터 분류
    #     for name, param in self.model.named_parameters():
    #         if name.startswith('vision_encoder') or 'vision_layernorm' in name:
    #             component_weights['vision'][name] = param  # 비전 레이어노름 포함
    #         elif name.startswith('qformer') or 'query_tokens' in name or 'extra_query_tokens' in name:
    #             component_weights['qformer'][name] = param  # 쿼리 토큰 포함
    #         elif 'project' in name.lower():
    #             component_weights['projection'][name] = param
    #         elif name.startswith('lm'):  # 기존 LLM 관련 파라미터
    #             component_weights['lm'][name] = param
    #         else:
    #             component_weights['misc'][name] = param

    #     buffer_weights = {name: buffer for name, buffer in self.model.named_buffers()}
    #     component_weights['qformer']['query_tokens'] = buffer_weights.pop('query_tokens', None)
    #     component_weights['qformer']['extra_query_tokens'] = buffer_weights.pop('extra_query_tokens', None)
        
    #     # 계층별 저장
    #     torch.save(
    #         {
    #             'params': component_weights['vision'],
    #             'buffers': {k: v for k, v in buffer_weights.items() if k.startswith('vision')}
    #         },
    #         os.path.join(save_dir, 'vision_components.pth')
    #     )
    #     print(f"vision_components.pth 저장 완료, {len(component_weights['vision'])}개 파라미터 저장")
        
    #     torch.save(
    #         {
    #             'params': component_weights['qformer'],
    #             'buffers': {k: v for k, v in buffer_weights.items() if k.startswith('qformer')},
    #             'special_tokens': {
    #                 'query_tokens': self.model.query_tokens,
    #                 'extra_query_tokens': self.model.extra_query_tokens
    #             }
    #         },
    #         os.path.join(save_dir, 'qformer_components.pth')
    #     )
    #     print(f"qformer_components.pth 저장 완료, {len(component_weights['qformer'])}개 파라미터 저장")
        
    #     torch.save(
    #         {
    #             'projection_params': component_weights['projection']
    #         },
    #         os.path.join(save_dir, 'projection_layers.pth')
    #     )
    #     print(f"projection_layers.pth 저장 완료, {len(component_weights['projection'])}개 파라미터 저장")
    #     print(f"lm_components.pth 저장 완료, {len(component_weights['lm'])}개 파라미터 확인")
    #     print(f"misc_components.pth 저장 완료, {len(component_weights['misc'])}개 파라미터 확인")
    #     print(f"misc 파라미터 이름 확인, {component_weights['misc'].keys()}")
        
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
                'do_sample': True,
                'max_new_tokens': 512,
            }
        )
        
        return response

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # VideoCaption 인스턴스 생성
    captioner = VideoCaption(current_dir)
    
    # # 비디오 경로 설정
    # media_path = os.path.join(current_dir, 'data', "본인이 원하는 비디오 경로")
    
    media_path = "/data/ephemeral/home/data/D3/DR/test/clips/D3_DR_0804_000048_001.mp4"
    
    # 캡션 생성
    caption = captioner.generate_caption(media_path, media_type='video')
    print("Generated Caption:", caption)

if __name__ == "__main__":
    main()