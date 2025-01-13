import os
from huggingface_hub import hf_hub_download

# 첫번째 로그인 설정 때만 주석 해제 후 사용해주세요
from huggingface_hub import login
login(token="your_token")

# 정확한 저장소와 파일 이름을 입력하면 다운로드 가능
model_name = [
    "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
    "OpenGVLab/InternVL"]

# 파일 이름 입력
file_name = [
    "InternVideo2-stage2_1b-224p-f4.pt",
    "internvl_c_13b_224px.pth"]

# 저장되는 파일 경로는 multi_modality/models/weights에 저장됨. 폴더가 없으면 생성되도록 os.makedirs 사용
local_dir = "../models/weights"
os.makedirs(local_dir, exist_ok=True)

for model, file in zip(model_name, file_name):
    model_file = hf_hub_download(
        repo_id=model,
        filename=file,
        local_dir=local_dir
    )
    print(f"모델 파일 다운로드 완료: {model_file}")
print("모든 파일 다운로드 완료")


