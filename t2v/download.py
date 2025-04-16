from huggingface_hub import snapshot_download
import os
# Official Repo for LongCLIP needed, plus the LongCLIP-L.pt from Huggingface

model_lists = ['OpenGVLab/InternVL3-8B', "Snowflake/snowflake-arctic-embed-l-v2.0"]
local_dirs = ['./weights/OpenGVLab/InternVL3-8B', './weights/weights_snowflake-arctic-embed-l-v2.0']
for model, local_dir in zip(model_lists, local_dirs):
    os.makedirs(local_dir, exist_ok=True)
    download_dir = snapshot_download(repo_id=model, local_dir=local_dir)