from huggingface_hub import snapshot_download
import os
model_name = "OpenGVLab/InternVL2_5-8B-MPO"
os.makedirs('./weights', exist_ok=True)
download_dir = snapshot_download(repo_id=model_name, local_dir='./weights' ,ignore_patterns=["*.py"])