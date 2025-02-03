import torch
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from model.utils.sampling import read_frames_cv2
from model_2_5.source.configuration_internvl_chat import InternVLChatConfig 
from model_2_5.source.modeling_internvl_chat_hico2 import InternVLChatModel

def load_model(config_path: str, model_path: str):
    """InternVideo2_5_Chat 모델과 토크나이저를 로드하는 함수"""
    config = InternVLChatConfig.from_pretrained(config_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InternVLChatModel.from_pretrained(model_path, config=config).to(device).half()
    
    return model, tokenizer, device

def preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """입력된 프레임을 224x224 크기로 Resize & 정규화하는 함수"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        frames = frames.float() / 255.0  # (0,1) 범위로 정규화
        T, C, H, W = frames.shape
        frames = frames.view(-1, C, H, W)  # (T, C, H, W) -> (T * C, H, W)
        frames = transform(frames)
        frames = frames.view(T, C, 224, 224)  # (T, C, 224, 224)
        
        return frames
    except Exception as e:
        raise RuntimeError(f"Error processing frames: {str(e)}")

def generate_response(model, tokenizer, device, video_path: str, question: str):
    """비디오 frame들 읽고 모델을 통해 문장을 생성하는 함수"""
    pixel_values, _, _ = read_frames_cv2(video_path=video_path, num_frames=8*48)
    pixel_values = preprocess_frames(pixel_values).to(device).half()
    
    generation_config = dict(
        do_sample=False,  # False = greedy search
        max_new_tokens=512,
        num_beams=1,  # Beam search
        temperature=0,
        top_p=0.1,
    )
    
    response = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question,
        generation_config=generation_config,
        num_patches_list=None
    )
    
    return response

if __name__ == "__main__":
    # 모델 및 토크나이저 로드
    config_path = "./model_2_5/configs/config.json"
    model_path = "OpenGVLab/InternVideo2_5_Chat_8B"
    model, tokenizer, device = load_model(config_path, model_path)
    
    # 입력 비디오 및 프롬프트 설정
    video_path = "/data/ephemeral/home/data_1/D3/DR/train/video/D3_DR_0804_000001.mp4"
    question = "Please describe the movements of the people in the video in temporal order."
    
    # 응답 생성
    response = generate_response(model, tokenizer, device, video_path, question)
    print(f"출력 결과: {response}")
