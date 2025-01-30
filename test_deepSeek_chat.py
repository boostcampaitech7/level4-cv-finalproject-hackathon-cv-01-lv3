# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "Qwen/Qwen2.5-7B-Instruct"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model_deepseek.sources.model_config import Qwen2_5_Config
from model_deepseek.sources.modeling_deepSeek_chat import DeepSeekChat
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np

class VideoCaption:
    def __init__(self, current_dir):
        # 설정 로드
        #self.config = Qwen2_5_Config.from_json_file(os.path.join(current_dir, "model_deepseek/configs/config.json"))
        self.config = Qwen2_5_Config.from_json_file("/data/ephemeral/home/deamin/level4-cv-finalproject-hackathon-cv-01-lv3/model_deepseek/configs/config.json")
        # model_path = os.path.join(current_dir, "model_deepseek/weights")
        model_path = "/data/ephemeral/home/deamin/level4-cv-finalproject-hackathon-cv-01-lv3/model_deepseek/weights"
        # 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_config.llm.pretrained_llm_path,
            trust_remote_code=True,
        )
        # self.lm # 
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

        from transformers import AutoModelForCausalLM
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # # 모델 초기화
        # if torch.cuda.is_available():
        #     self.model = DeepSeekChat.from_pretrained(
        #         model_path,
        #         config=self.config,
        #         torch_dtype=torch.bfloat16,
        #         trust_remote_code=True,
        #         use_auth_token=True
        #     ).cuda()
 
        # else:
        #     self.model = DeepSeekChat.from_pretrained(
        #         model_path,
        #         config=self.config,
        #         torch_dtype=torch.bfloat16,
        #         trust_remote_code=True
        #     )

        self.lm = self.lm.eval()
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
    
    def generate_caption(self, messages, media_path, media_type='video'):
        """비디오 캡션 생성"""
        
        # 채팅 히스토리 초기화
        chat_history = []
        
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"{messages}"}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.lm.device)
        
        generated_ids = self.lm.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # VideoCaption 인스턴스 생성
    captioner = VideoCaption(current_dir)
    
    # 비디오 경로 설정
    # media_path = os.path.join(current_dir, 'data', "본인이 원하는 비디오 경로")
    media_path = "/data/ephemeral/home/data/D3/DR/train/clips/D3_DR_0804_000001_001.mp4"
    # 캡션 생성
    caption = captioner.generate_caption(media_path, "What is the weather in Tokyo?", media_type='video')
    print("Generated Caption:", caption)

if __name__ == "__main__":
    main()