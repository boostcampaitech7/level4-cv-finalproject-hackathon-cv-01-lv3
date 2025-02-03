import os
import torch
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoConfig
import torch.nn.functional as F
import torchvision.transforms as T
from model.utils.data_utils_from_json import InternVideo2_VideoChat2_Dataset, InternVideo2_VideoChat2_DataLoader
from googletrans import Translator
import asyncio
import pandas as pd
from translate import translation
from model_2_5.source.configuration_internvl_chat import InternVLChatConfig 
from model_2_5.source.modeling_internvl_chat_hico2 import InternVLChatModel

def sec_to_time(sec: int) -> str:
    """
    초(sec)을 시:분:초로 변환할 수 있는 함수입니다.
    sec: 특정 시점의 초(sec)
    """
    s = sec % 60
    m = sec // 60
    h = sec // 3600
    return f"{h:02d}:{m:02d}:{s:02d}"

def inference_2_5(
    data_path: str,
    model_path: str = 'OpenGVLab/InternVideo2_5_Chat_8B',
    test_batch_size: int=1,
    test_num_workers: int=4,
    num_frames : int = 32,
    device: str='cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
    """
    InternVideo2 모델을 활용하여 Inference하는 함수입니다.
    ---------------------------------------------------
    args:
    data_path: 데이터가 있는 경로
    model_path: InternVideo2 모델이 있는 경로(fix)
    test_batch_size: Batch Size
    test_num_workers: num_workers 수 설정
    num_frames : 한 입력 비디오 및 segment의 sampling frame 개수
    device: cpu 혹은 cuda 등 Inference를 수행할 주체를 설정

    출력: 'segment_name', 'start_time', 'end_time', 'caption', 'caption_ko'로 이루어져 있는 v2t_submission.csv
    """
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    config = InternVLChatConfig.from_pretrained(
        os.path.join(current_dir, 'model_2_5', 'configs', 'config.json')
    )
    tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
                )

    # 모델 초기화
    model = InternVLChatModel.from_pretrained(
        model_path ,
        config =config
    ).cuda().half()

    
    test_dataset = InternVideo2_VideoChat2_Dataset(
        data_path=data_path,
        use_segment=True,
        use_audio=False,
        train=False,
        num_frames=num_frames ,
        save_frames_as_img=False
    )
    
    test_loader = InternVideo2_VideoChat2_DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=test_num_workers,
        shuffle=False,
        pin_memory=True,
        use_audio=False
    )
    model.eval()
    submission = pd.DataFrame(columns=['segment_name', 'start_time', 'end_time', 'caption', 'caption_ko'])
   

    for batch in test_loader:
        pixel_values= batch['frames'].to(device)
        #frames, frame_indices, int(total_frames/fps)
        pixel_values= pixel_values.squeeze(0) # B , T, C, H,W-> T , C , H,W 
        pixel_values = pixel_values.cuda().half()  # shape [16,3,224,224]
        print()
        # batch_chat : multiple quesion(no his)  , chat : single question
        outputs = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question="Describe this video in detail.",
            generation_config = {
                'do_sample': False, # false = greedy search , True 의 경우, t>0
                'max_new_tokens': 512,
                'num_beams' : 1, # bream search
                'temperature' : 0,
                'top_p' : 0.1, # top_k , p시 필요
            },
            history = None,
            return_history = False , 
            num_patches_list = None,
            )

        new_row = pd.DataFrame([
            {'segment_name': batch['segment_names'][0],
             'start_time': sec_to_time(batch['start_times'][0]), 
             'end_time': sec_to_time(batch['end_times'][0]), 
             'caption': outputs.strip(), 
             'caption_ko': asyncio.run(translation(outputs, 'en'))}])
        submission = pd.concat([submission, new_row], ignore_index=True)
        print(f"output : {outputs}") 
    submission.to_csv(f"v2t_submission.csv", index=False, encoding='utf-8')
    


def main():
    data_path = "../../data"

    inference_2_5(data_path)


if __name__ == "__main__":
    main()