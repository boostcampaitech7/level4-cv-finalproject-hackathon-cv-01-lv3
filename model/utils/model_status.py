import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model_config import VideoChat2Config
from modeling_videochat2 import InternVideo2_VideoChat2
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from train.utils.data_utils import InternVideo2_VideoChat2_Dataset, InternVideo2_VideoChat2_DataLoader


def model_status(model_path, config):
    """모델의 주요 구성 요소와 설정 상태를 확인하는 함수"""
    
    # 1. 모델 구성 요소 확인
    print("\n=== 모델 구성 요소 ===")
    print(f"Vision Encoder: {config.model_config.vision_encoder.name}")
    print(f"Bridge Type: {config.model_config.bridge.name}")
    print(f"LLM: {config.model_config.llm.name}")
    
    # 2. 모델 초기화
    model = InternVideo2_VideoChat2.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 3. Frozen 상태 확인
    print("\n=== Frozen 상태 ===")
    print(f"Vision Encoder Frozen: {model.freeze_vision_encoder}")
    print(f"Bridge (Q-Former) Frozen: {model.freeze_bridge}")
    print(f"LLM Frozen: {model.freeze_llm}")
    
    # 4. 학습 가능한 파라미터 확인
    total_params = 0
    trainable_params = 0
    
    print("\n=== 파라미터 상태 ===")
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            print(f"학습 가능한 레이어: {name}")
    
    print(f"\n전체 파라미터: {total_params:,}")
    print(f"학습 가능한 파라미터: {trainable_params:,}")
    print(f"학습 가능한 파라미터 비율: {100 * trainable_params / total_params:.2f}%")
    
    # 5. LoRA 설정 확인
    if hasattr(model, 'use_lora') and model.use_lora:
        print("\n=== LoRA 설정 ===")
        print(f"LoRA Rank (r): {config.model_config.llm.lora_r}")
        print(f"LoRA Alpha: {config.model_config.llm.lora_alpha}")
        print(f"LoRA Dropout: {config.model_config.llm.lora_dropout}")
    
    model.use_lora = False
    
    # 5. LoRA 설정 확인
    if hasattr(model, 'use_lora') and model.use_lora:
        print("\n=== LoRA 설정, False 이후 ===")
        print(f"LoRA Rank (r): {config.model_config.llm.lora_r}")
        print(f"LoRA Alpha: {config.model_config.llm.lora_alpha}")
        print(f"LoRA Dropout: {config.model_config.llm.lora_dropout}")
    


    # 6. 메모리 사용량 확인 (CUDA 사용 시)
    if torch.cuda.is_available():
        print("\n=== CUDA 메모리 사용량 ===")
        print(f"할당된 메모리: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"캐시된 메모리: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    return model

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = VideoChat2Config.from_json_file(os.path.join(current_dir, 'config.json'))
    model = model_status(current_dir, config)
    return model

if __name__ == "__main__":
    main()