from utils.video_load import VideoLoad
import os
import sys
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from model_2_5.source.configuration_internvl_chat import InternVLChatConfig 

from model.utils.sampling import read_frames_cv2

from transformers import AutoModel, AutoTokenizer


class InternVideo2_5(VideoLoad):
    def __init__(self):
        super().__init__()
        self.internvideo_2_5_config = InternVLChatConfig.from_pretrained(os.path.join(project_root, "model/configs/config_internVideo2_5.json"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = "OpenGVLab/InternVideo2_5_Chat_8B"
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
        ).to(self.device).half().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
    def generate_video_summary(self, video_path):
        pixel_values, _, _ = read_frames_cv2(video_path=video_path, num_frames= 16)
        pixel_values = self.preprocess_frames(pixel_values).to(self.device).half()

        pixel_values= pixel_values.squeeze(0) # B , T, C, H,W-> T , C , H,W 
        pixel_values = pixel_values.cuda().half()  # shape [16,3,224,224]
        
        summary = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=(
            "Carefully watch the video and pay attention to the cause and sequence of events, "
            "the detail and movement of objects, and the action and pose of persons.\n\n"
            "Then, perform the following tasks:\n"
            "1.**Identify the primary genre** of the video from the following categories: "
            "[Movie, Drama, Horror, Sci-Fi, Fantasy, Sport, News]. "
            "Provide a brief justification for your choice based on visual elements, character actions, setting, and overall tone.\n\n"
            "2.**Summarize the video** in 2-3 sentences, concisely capturing the core narrative, key events, and main character interactions. "
            "Ensure the summary is clear and logically structured, focusing on the most important aspects of the video.\n\n"
            "Return the result in the following JSON format:\n"
            "{\n"
            '    "Genre": "<Identified genre>",\n'
            '    "Summary": "<Concise video summary>"\n'
            "}"
        ),
        generation_config = {
            'do_sample': False, # False is greedy search
            'max_new_tokens': 512,
            'num_beams' : 1, 
            'temperature' : 0,
            'top_p' : 0.1,
            'top_k' : None
        },
        return_history = False,
        history = None,
        num_patches_list = None,        
        )
        return summary

    def generate_video_summary(self, video_path):
        pixel_values, _, _ = read_frames_cv2(video_path=video_path, num_frames= 16)
        pixel_values = self.preprocess_frames(pixel_values).to(self.device).half()

        pixel_values= pixel_values.squeeze(0) # B , T, C, H,W-> T , C , H,W 
        pixel_values = pixel_values.cuda().half()  # shape [16,3,224,224]

        summary = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=(
            "Carefully watch the video and pay attention to the cause and sequence of events, "
            "the detail and movement of objects, and the action and pose of persons.\n\n"
            "Then, perform the following tasks:\n"
            "1.**Identify the primary genre** of the video from the following categories: "
            "[Movie, Drama, Horror, Sci-Fi, Fantasy, Sport, News]. "
            "Provide a brief justification for your choice based on visual elements, character actions, setting, and overall tone.\n\n"
            "2.**Summarize the video** in 2-3 sentences, concisely capturing the core narrative, key events, and main character interactions. "
            "Ensure the summary is clear and logically structured, focusing on the most important aspects of the video.\n\n"
            "Return the result in the following JSON format:\n"
            "{\n"
            '    "Genre": "<Identified genre>",\n'
            '    "Summary": "<Concise video summary>"\n'
            "}"
        ),
        generation_config = {
            'do_sample': False, # False is greedy search
            'max_new_tokens': 512,
            'num_beams' : 1, 
            'temperature' : 0,
            'top_p' : 0.1,
            'top_k' : None
        },
        return_history = False,
        history = None,
        num_patches_list = None,
        )
        return summary
        

