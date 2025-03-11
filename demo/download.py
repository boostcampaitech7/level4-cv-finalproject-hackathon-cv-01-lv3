from huggingface_hub import snapshot_download
import os
# Official Repo for LongCLIP needed, plus the LongCLIP-L.pt from Huggingface

tt_model_name = "snowflake/snowflake-arctic-embed-l-v2.0"
local_dir = '../t2v/weights/weights_snowflake-arctic-embed-l-v2.0'
os.makedirs(local_dir, exist_ok=True)
download_dir = snapshot_download(repo_id=tt_model_name, local_dir=local_dir)