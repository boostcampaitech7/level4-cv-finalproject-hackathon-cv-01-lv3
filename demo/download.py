from huggingface_hub import snapshot_download
import os
model_name = "yibinlei/LENS-d8000"
local_dir = './weights_yibinlei/LENS-d8000'
os.makedirs(local_dir, exist_ok=True)
download_dir = snapshot_download(repo_id=model_name, local_dir=local_dir)# ,ignore_patterns=["*.py"])