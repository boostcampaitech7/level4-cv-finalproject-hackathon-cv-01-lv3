import os
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel

# 첫번째 로그인 설정 때만 주석 해제 후 사용해주세요
from huggingface_hub import login
login(token="your_token")

# def download_model(model_list:list, file_list:list, local_dir:str) -> None:
#     '''
#     모델을 다운받을 수 있는 함수,
    
#     Args:
#         model_list(list): 모델 이름 리스트(허깅페이스 기준)
#         file_list(list): 파일 이름 리스트
#             - 특정 파일을 다운 받을 경우 model_list와 같은 위치에 입력
#             - 전체 폴더를 다운 받아 모델로 정의할 경우 공백 문자열 입력
#         local_dir(str): 다운받을 경로
    
#     Description:
#         weights 폴더에 .pt/.pth 파일 저장
    
#     Returns:
#         None
#     '''
#     for i, model_name in enumerate(model_list):
#         if model_name == "google-bert/bert-large-uncased":
#             tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", cache_dir=local_dir)
#             model = AutoModel.from_pretrained("bert-large-uncased", cache_dir=local_dir)
#             print(f"모델 파일 다운로드 완료: {model_name}")
#         else:
#             model = hf_hub_download(
#                 repo_id=model_name,
#                 filename=file_list[i],
#                 local_dir=local_dir
#             )
#             print(f"모델 파일 다운로드 완료: {model}")
#     print("모든 파일 다운로드 완료")

# def main():
#     # 모델 이름 입력
#     model_name = [
#         "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
#         "OpenGVLab/InternVL",
#         "google-bert/bert-large-uncased"]

#     # 파일 이름 입력
#     file_name = [
#         "InternVideo2-stage2_1b-224p-f4.pt",
#         "internvl_c_13b_224px.pth",
#         ""]

#     # 저장되는 파일 경로, 폴더가 없으면 생성되도록 os.makedirs 사용
#     local_dir = "../models/weights"
#     os.makedirs(local_dir, exist_ok=True)

#     download_model(model_name, file_name, local_dir)

# if __name__ == "__main__":
#     main()