---
language:
- en
library_name: transformers
license: apache-2.0
metrics:
- accuracy
tags:
- multimodal
pipeline_tag: video-text-to-text
model-index:
- name: InternVideo2.5
  results:
  - task:
      type: multimodal
    dataset:
      name: MLVU
      type: mlvu
    metrics:
    - type: accuracy
      value: 72.8
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MVBench
      type: mvbench
    metrics:
    - type: accuracy
      value: 75.7
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: Perception Test
      type: percepTest
    metrics:
    - type: accuracy
      value: 74.9
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: LongVideoBench
      type: longvideobench
    metrics:
    - type: accuracy
      value: 60.6
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: VideoMME (w/o sub)
      type: videomme
    metrics:
    - type: accuracy
      value: 65.1
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: LVBench
      type: lvbench
    metrics:
    - type: accuracy
      value: 46.4
      name: accuracy
      verified: true


---

# 📕InternVideo2.5⚡
<!-- [\[📰 Blog\]](https://internvideo.github.io/blog/2024-12-31-VideoChat-Flash) -->
[\[📂 GitHub\]](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2.5)  
[\[📜 Tech Report\]](https://arxiv.org/abs/2501.12386) 
<!-- [\[🗨️ Chat Demo\]](https://huggingface.co/spaces/OpenGVLab/VideoChat-Flash) -->

 InternVideo2.5 is a video multimodal large language model (MLLM, built upoon InternVL2.5) enhanced with **long and rich context (LRC) modeling**. It significantly improves upon existing MLLMs by enhancing their ability to perceive fine-grained details and capture long-form temporal structures. We achieve this through dense vision task annotations using direct preference optimization (TPO) and compact spatiotemporal representations via adaptive hierarchical token compression (HiCo).




## 📈 Performance
| Model |  MVBench | LongVideoBench |  VideoMME(w/o sub)| 
| ---   |  ---     |   ---            | ---     | 
|InternVideo2.5| 75.7 |  60.6   | 65.1| 

## 🚀 How to use the model

First, you need to install [flash attention2](https://github.com/Dao-AILab/flash-attention) and some other modules. We provide a simple installation example below:
```
pip install transformers==4.40.1
pip install av
pip install imageio
pip install decord
pip install opencv-python
pip install flash-attn --no-build-isolation
```
Then you could use our model:
```python
from transformers import AutoModel, AutoTokenizer

# model setting
model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
image_processor = model.get_vision_tower().image_processor


# evaluation setting
max_num_frames = 512
generation_config = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.1,
    num_beams=1
)

video_path = "your_video.mp4"

# single-turn conversation
question1 = "Describe this video in detail."
output1, chat_history = model.chat(video_path=video_path, tokenizer=tokenizer, user_prompt=question1, return_history=True, max_num_frames=max_num_frames, generation_config=generation_config)

print(output1)

# multi-turn conversation
question2 = "How many people appear in the video?"
output2, chat_history = model.chat(video_path=video_path, tokenizer=tokenizer, user_prompt=question2, chat_history=chat_history, return_history=True, max_num_frames=max_num_frames, generation_config=generation_config)

print(output2)
```

## ✏️ Citation

```bibtex

@article{wang2025internvideo,
  title={InternVideo2.5: Empowering Video MLLMs with Long and Rich Context Modeling},
  author={Wang, Yi and Li, Xinhao and Yan, Ziang and He, Yinan and Yu, Jiashuo and Zeng, Xiangyu and Wang, Chenting and Ma, Changlian and Huang, Haian and Gao, Jianfei and Dou, Min and Chen, Kai and Wang, Wenhai and Qiao, Yu and Wang, Yali and Wang, Limin},
  journal={arXiv preprint arXiv:2501.12386},
  year={2025}
}
```