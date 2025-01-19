import os
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel

# 첫번째 로그인 설정 때만 주석 해제 후 사용해주세요
from huggingface_hub import login
login(token="your_token")

def download_model(model_list:list, file_list:list, local_dir:str) -> None:
    '''
    모델을 다운받을 수 있는 함수,
    
    Args:
        model_list(list): 모델 이름 리스트(허깅페이스 기준)
        file_list(list): 파일 이름 리스트
            - 특정 파일을 다운 받을 경우 model_list와 같은 위치에 입력
            - 전체 폴더를 다운 받아 모델로 정의할 경우 공백 문자열 입력
        local_dir(str): 다운받을 경로
    
    Description:
        weights 폴더에 .pt/.pth 파일 저장
    
    Returns:
        None
    '''
    for i in range(len(file_list)):
        model = hf_hub_download(
            repo_id=model_list[0],
            filename=file_list[i],
            local_dir=local_dir
        )
        print(f"모델 파일 다운로드 완료: {file_list[i]}")
    print("모든 파일 다운로드 완료")

def main():
    # 모델 이름 입력
    model_name = [
        "OpenGVLab/InternVideo2-Chat-8B",
        ]

    # 파일 이름 입력
    file_name = [
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer.model.v3",
        "tokenizer_config.json",
        "config.json",
        ]

    # 현재 파일의 디렉토리 경로를 가져옵니다
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(current_dir, exist_ok=True)

    download_model(model_name, file_name, current_dir)

if __name__ == "__main__":
    main()