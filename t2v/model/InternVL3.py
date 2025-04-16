import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
import pandas as pd

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

if __name__ == "__main__":

    vision_model = AutoModel.from_pretrained(
        '../weights/OpenGVLab/VisionExtractionVL3',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True)#.cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained('../weights/OpenGVLab/VisionExtractionVL3', trust_remote_code=True, use_fast=False)

    for name, param in vision_model.named_parameters():
        print(f"{name}: {param.shape}")

    # df = pd.read_csv("./new_testset_with_generated_query.csv")
    # target = df[:5]
    # videos = []
    # video_paths = [(f"../../YT8M/clips/{video['segment_name']}.mp4", video['Full Video Description']) for _, video in target.iterrows()]
    # for video_path, caption in video_paths:
    #     pixel_values, num_patches_list = load_video(video_path, num_segments=16, max_num=1)
    #     videos.append((pixel_values.to(torch.bfloat16), caption))
    # ###########################

    # from text_model import Snowflake
    # import torch.nn as nn
    # import torch.nn.functional as F

    # text_model = Snowflake()

    # class ProjectionHead(nn.Module):
    #     def __init__(self, in_dim, out_dim):
    #         super().__init__()
    #         self.fc = nn.Linear(in_dim, out_dim)

    #     def forward(self, x):
    #         return F.normalize(self.fc(x), dim=-1)

    # class VideoTextRetrievalModel(nn.Module):
    #     def __init__(self, vision_dim, text_dim, embed_dim):
    #         super().__init__()
    #         self.vision_proj = ProjectionHead(vision_dim, embed_dim)
    #         self.text_proj = ProjectionHead(text_dim, embed_dim)

    #     def forward(self, vision_feat, text_feat):
    #         text_feat = torch.from_numpy(text_feat).to(torch.bfloat16).cuda()
    #         vision_feat = vision_feat.to(torch.bfloat16).cuda()
    #         vision_embeds = self.vision_proj(vision_feat)        # [B, embed_dim]
    #         text_embeds = self.text_proj(text_feat)              # [B, embed_dim]
    #         return vision_embeds, text_embeds

    # def compute_contrastive_loss(vision_embeds, text_embeds, temperature=0.07):
    #     logits = torch.matmul(vision_embeds, text_embeds.T) / temperature
    #     labels = torch.arange(len(vision_embeds)).to(vision_embeds.device)
    #     print(f"labels.shape: {labels.shape}, {labels.dtype}")

    #     loss_i2t = F.cross_entropy(logits, labels)
    #     loss_t2i = F.cross_entropy(logits.T, labels)
    #     return (loss_i2t + loss_t2i) / 2


    # model = VideoTextRetrievalModel(vision_dim=3584, text_dim=1024, embed_dim=1024).to(torch.bfloat16).cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # for epoch in range(100):
    #     for pixel_values, text_queries in videos:
    #         with torch.no_grad():
    #             vision_feats = vision_model.extract_feature(pixel_values.cuda())  # [B, 256, 3584]
    #             vision_feats = vision_feats.mean(dim=1)  # Temporal 평균 → [B, 3584]

    #             text_feats = text_model.encode_text(text_queries)

    #         vision_embeds, text_embeds = model(vision_feats, text_feats)
    #         loss = compute_contrastive_loss(vision_embeds, text_embeds)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         print(f"Loss: {loss.item():.4f}")