import torch
import torch.nn as nn
import torch.nn.functional as F
from text_model import Snowflake
from transformers import AutoModel, AutoTokenizer
from aggregator import SpaceTimeBridgeEncoder

class AlignModel(nn.Module):
    def __init__(self, 
                 video_dim, 
                 text_dim, 
                 projection_dim,
                 load_checkpoint=None):
        super().__init__()

        self.video_dim = video_dim
        self.text_dim = text_dim
        self.text_model = Snowflake()
        self.video_model = AutoModel.from_pretrained(
        '../weights/OpenGVLab/VisionExtractionVL3',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True).cuda().eval()
        self.bridge = SpaceTimeBridgeEncoder(transformer_dim=3584).to(torch.bfloat16)

        self.txt_proj = nn.Sequential(nn.ReLU(),
                                 nn.Linear(self.text_dim, projection_dim))
        
        self.vid_proj = nn.Sequential(nn.Linear(self.video_dim, projection_dim))
    
    def forward(self, data, return_embeds=True):
        if len(data) == 0:
            raise RuntimeError(f"No Data in data: {data}. It should have dict structure with keys 'text' and 'video'")
        if isinstance(data, list):
            text_data = [x['text'] for x in data]
            video_data = [x['video'] for x in data]
        else:
            text_data = [data['text']]
            video_data = [data['video']]

        text_embeddings = self.compute_text(text_data)
        video_embeddings = self.compute_video(video_data)

        if return_embeds:
            return text_embeddings, video_embeddings

        return sim_matrix(text_embeddings, video_embeddings)
    
    def _batch_encode_text(self, text_data):
        text_embeddings = []
        for txt in text_data:
            text_embedding = self.text_model.encode_text(txt).to(torch.bfloat16)
            text_embeddings.append(text_embedding.cpu())
        return torch.stack(text_embeddings, dim=0)
    
    def _batch_encode_video(self, video_data):
        video_embeddings = []
        for vid in video_data:
            with torch.no_grad():  # 추가 필요
                video_embedding = self.video_model.extract_feature(vid.cuda())
            video_embeddings.append(video_embedding.cpu())
        return torch.stack(video_embeddings, dim=0)
    
    def compute_text(self, text_data):
        text_embeddings = self._batch_encode_text(text_data).detach().cuda()
        print(f"After Passing Snowflake Encode_text:: text_embeddings.shape: {text_embeddings.shape}")
        text_embeddings = self.txt_proj(text_embeddings.cuda())
        print(f"After Passing Projection:: text_embeddings.shape: {text_embeddings.shape}")
        return text_embeddings
    
    def compute_video(self, video_data):
        video_embeddings = self._batch_encode_video(video_data).detach()
        print(f"After Passing InternViT's extract_features:: video_embeddings.shape: {video_embeddings.shape}")
        video_embeddings = self.bridge(video_embeddings.cuda())
        print(f"After Passing Bridge Model:: video_embeddings.shape: {video_embeddings.shape}")
        video_embeddings = self.vid_proj(video_embeddings)
        print(f"After Passing Projection:: video_embeddings.shape: {video_embeddings.shape}")
        return video_embeddings


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def compute_similarity(a, b, eps=1e-8):
    sim = sim_matrix(a,b,eps=eps)
    return sim, sim.t()



if __name__ == '__main__':
    import pandas as pd
    from InternVL3 import load_video
    df = pd.read_csv("../../new_testset_with_generated_query.csv")
    target = df[:30]
    videos = []
    video_paths = [(f"../../../YT8M/clips/{video['segment_name']}.mp4", video['Full Video Description']) for _, video in target.iterrows()]
    for video_path, caption in video_paths:
        pixel_values, num_patches_list = load_video(video_path, num_segments=16, max_num=1)
        videos.append({'video': pixel_values.to(torch.bfloat16), 'text': caption})
    
    model = AlignModel(3584, 1024, 256).type(torch.bfloat16).cuda()
    text_embedding, video_embedding = model(videos[:30])
    print(f"text_embedding: {text_embedding.shape}")
    print(f"video_embedding: {video_embedding.shape}")
    print(f"sim_matrix: {sim_matrix(text_embedding, video_embedding)}")