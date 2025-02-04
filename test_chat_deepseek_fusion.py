from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 토크나이저 패딩 설정 추가
tokenizer.padding_side = "left"

prompt = """다음 멀티모달 비디오 캡션에 대한 퓨전 캡션을 생성해주세요. 아래에 적힌 캡션 설명과 예시를 참고하세요.

# 1. Audio: 배경 음향 및 주요 소리 요소 기술
# 2. Visual: 프레임별 주요 객체 및 움직임 기술 
# 3. Fusion: 오디오-비주얼 정보를 종합한 자연스러운 설명 생성

# 예시 (밤하늘 비디오):
# [Audio Caption] 
# - 귀뚜라미 울음소리와 약한 바람 소리
# - 가끔 들리는 야생동물 발소리
# - 먼 곳에서 들리는 차량 엔진 음

# [Visual Caption]
# - 구름 없는 밤하늘에 수많은 별들 반짝임
# - 은하수 띠가 수평선 위로 뚜렷하게 관측됨
# - 간헐적으로 유성우가 수직으로 하강

# [Fusion Caption]
# "고요한 밤, 귀뚜라미 울음소리가 밤공기를 가르는 가운데, 구름 한 점 없는 광활한 하늘에는 은하수의 장관이 펼쳐집니다. 
  저 멀리 풀슾에 야생돌물의 발소리와 귀뚜라미 울음소리가 들리고, 세워둔 차량의 엔진 소리가 들리네요.
  수시로 떨어지는 유성우가 야생의 정취를 더하는 이 풍경은 참으로 아름답네요.

이제 다음 비디오에 대해 [fusion caption] 을 작성하세요. 한글로 작성하세요.

[Video Description]
화창한 해변에서 파도가 밀려오는 풍경

[Audio Caption]
- 규칙적인 파도 부서지는 소리
- 갈매기 울음소리
- 약한 바람 소리

[Visual Caption]
- 푸른 하늘과 에메랄드빛 바다
- 하얀 거품을 일으키며 해변을 적시는 파도
- 상공을 선회하는 갈매기 무리

[Fusion 평가 포인트]
오디오-비주얼 요소의 시간적 동기화 정확도
감각 정보의 자연스러운 통합 여부
"""

# prompt = """Please generate multimodal captions for the following video. Follow these steps:

# Example (Night sky video):
# [Sound Caption]
# - Cricket chirps with faint wind sounds
# - Occasional wildlife footsteps
# - Distant vehicle engine noises

# [Visual Caption]
# - Countless stars twinkling in cloudless night sky
# - Distinct Milky Way band visible above horizon
# - Intermittent meteor showers vertically descending

# [Fused Caption]
# "Under the serene mountain night air pierced by cricket songs, the crystal-clear sky unveils a galactic spectacle. Frequent meteor showers enhance the wilderness atmosphere in this breathtaking scene..."

# Now apply the same format to the following video:

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 안전한 generation config 사용
generation_config = model.generation_config
generation_config.update(
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.5
)

# 오류 처리 추가
try:
    generated_ids = model.generate(
        **model_inputs,
        generation_config=generation_config
    )
except Exception as e:
    print(f"생성 오류 발생: {str(e)}")
    exit(1)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 출력 처리 개선
response = tokenizer.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True  # 불필요한 공백 제거
)[0].strip()  # 앞뒤 공백 정리

print(response)
    
# import os
# import torch
# from transformers import AutoTokenizer, AutoConfig
# from model_deepseek.sources.model_config import Qwen2_5_Config
# from model_deepseek.sources.modeling_deepSeek_chat import DeepSeekChat
# from decord import VideoReader, cpu
# import torch.nn.functional as F
# import torchvision.transforms as T
# from PIL import Image
# import numpy as np

# class VideoCaption:
#     def __init__(self, current_dir):
#         # 설정 로드
#         #self.config = Qwen2_5_Config.from_json_file(os.path.join(current_dir, "model_deepseek/configs/config.json"))
#         self.config = Qwen2_5_Config.from_json_file("/data/ephemeral/home/deamin/level4-cv-finalproject-hackathon-cv-01-lv3/model_deepseek/configs/config.json")
#         # model_path = os.path.join(current_dir, "model_deepseek/weights")
#         model_path = "/data/ephemeral/home/deamin/level4-cv-finalproject-hackathon-cv-01-lv3/model_deepseek/weights"
#         # 토크나이저 초기화
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.config.model_config.llm.pretrained_llm_path,
#             trust_remote_code=True,
#         )
#         # self.lm # 
#         model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

#         from transformers import AutoModelForCausalLM
#         self.lm = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype="auto",
#             device_map="auto"
#         )
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         # # 모델 초기화
#         # if torch.cuda.is_available():
#         #     self.model = DeepSeekChat.from_pretrained(
#         #         model_path,
#         #         config=self.config,
#         #         torch_dtype=torch.bfloat16,
#         #         trust_remote_code=True,
#         #         use_auth_token=True
#         #     ).cuda()
 
#         # else:
#         #     self.model = DeepSeekChat.from_pretrained(
#         #         model_path,
#         #         config=self.config,
#         #         torch_dtype=torch.bfloat16,
#         #         trust_remote_code=True
#         #     )

#         self.lm = self.lm.eval()
#     def load_media(self, media_path, media_type='video', num_segments=8, resolution=224, hd_num=6):
#         """비디오 로드 및 전처리"""
#         if media_type == 'video':
#             vr = VideoReader(media_path, ctx=cpu(0))

#             # 균일한 간격으로 프레임 인덱스 추출
#             num_frames = len(vr)
#             indices = self._get_frame_indices(num_frames, num_segments)
            
#             # 프레임 추출 및 전처리
#             frames = vr.get_batch(indices).asnumpy()
#             # NumPy 배열을 PyTorch Tensor로 변환
#             frames = torch.from_numpy(frames)
#             frames = frames.permute(0, 3, 1, 2)  # (N, C, H, W)
            
#             # 정규화
#             frames = self._transform_frames(frames, resolution)
            
#             return frames.unsqueeze(0)  # 배치 차원 추가

#         elif media_type == 'image':
#             image = Image.open(media_path)
#             image = image.resize((resolution, resolution))
#             image = np.array(image)
#             image = torch.from_numpy(image)
#             image = image.permute(2, 0, 1)  # (C, H, W)
#             image = image.unsqueeze(0)  # 배치 차원 추가
#             return image

#     def _get_frame_indices(self, num_frames, num_segments):
#         """균일한 간격으로 프레임 인덱스 추출"""
#         seg_size = float(num_frames - 1) / num_segments
#         indices = torch.linspace(0, num_frames-1, num_segments).long()
#         return indices

#     def _transform_frames(self, frames, resolution):
#         """프레임 전처리"""
#         mean = torch.tensor([0.485, 0.456, 0.406])
#         std = torch.tensor([0.229, 0.224, 0.225])
        
#         transform = T.Compose([
#             T.Resize((resolution, resolution)),
#             T.Lambda(lambda x: x.float().div(255.0)),
#             T.Normalize(mean, std)
#         ])
        
#         return transform(frames)
    
#     def generate_caption(self, messages, media_path, media_type='video'):
#         """비디오 캡션 생성"""
        
#         # 채팅 히스토리 초기화
#         chat_history = []
        
#         messages = [
#             {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#             {"role": "user", "content": f"{messages}"}
#         ]
#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         model_inputs = self.tokenizer([text], return_tensors="pt").to(self.lm.device)
        
#         generated_ids = self.lm.generate(
#             input_ids=model_inputs.input_ids,
#             attention_mask=model_inputs.attention_mask,
#             max_new_tokens=512,
#             pad_token_id=self.tokenizer.eos_token_id
#         )
#         generated_ids = [
#             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#         ]

#         response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#         return response
    

# def main():
#     current_dir = os.path.dirname(os.path.abspath(__file__))

#     # VideoCaption 인스턴스 생성
#     captioner = VideoCaption(current_dir)
    
#     # 비디오 경로 설정
#     # media_path = os.path.join(current_dir, 'data', "본인이 원하는 비디오 경로")
#     media_path = "/data/ephemeral/home/data/D3/DR/train/clips/D3_DR_0804_000001_001.mp4"
#     # 캡션 생성
#     caption = captioner.generate_caption(media_path, "What is the weather in Tokyo?", media_type='video')
#     print("Generated Caption:", caption)

# if __name__ == "__main__":
#     main()