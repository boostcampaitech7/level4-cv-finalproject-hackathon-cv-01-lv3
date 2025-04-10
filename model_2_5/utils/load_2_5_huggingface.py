from huggingface_hub import snapshot_download
import os
import shutil

# 저장할 디렉토리 설정
current_dir = "../"
config_dir = os.path.join(current_dir, "configs")
weights_dir = os.path.join(current_dir, "weights")

# 디렉토리 생성
os.makedirs(config_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

# Hugging Face 모델 다운로드
model_name = "OpenGVLab/InternVideo2_5_Chat_8B"
download_dir = snapshot_download(repo_id=model_name, local_dir=weights_dir ,ignore_patterns=["*.py"])

# 파일 이동
for root, _, files in os.walk(download_dir):
    for file in files:
        file_path = os.path.join(root, file)

        # .py 파일은 건너뛴다
        if file.endswith(".py"):
            continue

        # config.json은 configs 폴더로 이동
        if file == "config.json":
            shutil.move(file_path, os.path.join(config_dir, file))

