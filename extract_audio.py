import ffmpeg
import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model.sources.model_config import VideoChat2Config
from model.sources.modeling_videochat2 import InternVideo2_VideoChat2
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import whisper
import glob
import json  # JSON 저장을 위한 모듈 추가
import time
from data.utils.load_video import sec_to_time
movie_path = "/data/ephemeral/home/test/YT8M/Movieclips/origin"

class VideoCaption:
    def __init__(self, current_dir):
        # 설정 로드
        self.config = VideoChat2Config.from_json_file(os.path.join(current_dir, "model/configs/config.json"))

        # model_path = os.path.join(current_dir, "model/weights")
        model_path = "/data/ephemeral/home/hongjoo/level4-cv-finalproject-hackathon-cv-01-lv3/model/weights"
    
        # 토크나이저 초기화 (Mistral-7B)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_config.llm.pretrained_llm_path,
            trust_remote_code=True,
            use_fast=False,
            token=os.getenv('HF_TOKEN')
        )

        # 모델 초기화
        if torch.cuda.is_available():
            self.model = InternVideo2_VideoChat2.from_pretrained(
                model_path,
                config=self.config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).cuda()
 
        else:
            self.model = InternVideo2_VideoChat2.from_pretrained(
                model_path,
                config=self.config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        self.model.eval()

    def load_media(self, media_path, media_type='video', num_segments=8, resolution=224, hd_num=6):
        """비디오 로드 및 전처리"""
        if media_type == 'video':
            vr = VideoReader(media_path, ctx=cpu(0))

            # 균일한 간격으로 프레임 인덱스 추출
            num_frames = len(vr)
            indices = self._get_frame_indices(num_frames, num_segments)
            
            # 프레임 추출 및 전처리
            frames = vr.get_batch(indices).asnumpy()
            # NumPy 배열을 PyTorch Tensor로 변환
            frames = torch.from_numpy(frames)
            frames = frames.permute(0, 3, 1, 2)  # (N, C, H, W)
            
            # 정규화
            frames = self._transform_frames(frames, resolution)
            
            return frames.unsqueeze(0)  # 배치 차원 추가

        elif media_type == 'image':
            image = Image.open(media_path)
            image = image.resize((resolution, resolution))
            image = np.array(image)
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)  # (C, H, W)
            image = image.unsqueeze(0)  # 배치 차원 추가
            return image

    def _get_frame_indices(self, num_frames, num_segments):
        """균일한 간격으로 프레임 인덱스 추출"""
        seg_size = float(num_frames - 1) / num_segments
        indices = torch.linspace(0, num_frames-1, num_segments).long()
        return indices

    def _transform_frames(self, frames, resolution):
        """프레임 전처리"""
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        transform = T.Compose([
            T.Resize((resolution, resolution)),
            T.Lambda(lambda x: x.float().div(255.0)),
            T.Normalize(mean, std)
        ])
        
        return transform(frames)

    def generate_caption(self, media_path, media_type='video'):
        """비디오 캡션 생성"""

        # 비디오 로드 및 전처리
        if media_type == 'video':
            video_tensor = self.load_media(
                media_path, 
                media_type=media_type,
                num_segments=self.config.model_config.vision_encoder.num_frames
            )
        
        elif media_type == 'image':
            video_tensor = self.load_media(
                media_path, 
                media_type=media_type,
                num_segments=1
            )
        
        elif media_type == 'video_audio':
            video_tensor = self.load_media(
                media_path, 
                media_type=media_type,
                num_segments=self.config.model_config.vision_encoder.num_frames
            )
        
        if torch.cuda.is_available():
            video_tensor = video_tensor.cuda()

        # 채팅 히스토리 초기화
        chat_history = []
        
        # 캡션 생성
        response, _ = self.model.chat(
            tokenizer=self.tokenizer,
            msg='',
            user_prompt='Describe the video step by step',
            instruction="Carefully watch the video and describe what is happening in detail.",
            media_type=media_type,
            media_tensor=video_tensor,
            chat_history=chat_history,
            return_history=True,
            generation_config={
                'do_sample': False,
                'max_new_tokens': 512,
            }
        )
        
        return response

def cap_fusion():
    # 1. 비전 캡션 파일 목록 로드
    vision_caption_dir = '/data/ephemeral/home/test/YT8M/Movieclips/labels'
    speech_caption_dir = '/data/ephemeral/home/test/YT8M/Movieclips/origin/stt'
    
    vision_files = glob.glob(os.path.join(vision_caption_dir, '*.json'))
    
    for vision_path in vision_files:
        # 2. 원본 비디오 ID 추출 (예: yt8m_Movieclips_xcJXT5lc1Bg)
        base_video_id = '_'.join(os.path.basename(vision_path).split('_')[:-1])
        speech_path = os.path.join(speech_caption_dir, f"{base_video_id}.json")
        
        if not os.path.exists(speech_path):
            print(f"음성 캡션 없음: {speech_path}")
            continue
            
        # 3. 캡션 통합 실행
        try:
            merged_path = merge_captions(vision_path, speech_path)
            print(f"생성된 통합 파일: {merged_path}")
        except Exception as e:
            print(f"에러 발생: {vision_path} - {str(e)}")

def time_to_seconds(time_str):
    """HH:MM:SS 형식을 초 단위로 변환"""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def merge_captions(vision_caption_path, speech_caption_path):
    # 비전 캡션 로드
    with open(vision_caption_path, 'r') as f:
        vision_data = json.load(f)
    
    # 음성 캡션 로드
    with open(speech_caption_path, 'r') as f:
        speech_data = json.load(f)

    # 비전 타임라인 추출
    video_id = next(iter(vision_data))  # 첫 번째 키 추출 (e.g. "yt8m_Movieclips_xcJXT5lc1Bg_001")
    vision_start = time_to_seconds(vision_data[video_id]['start_time'])
    vision_end = time_to_seconds(vision_data[video_id]['end_time'])

    # 시간대 필터링
    overlapping_speech = []
    for speech in speech_data:
        speech_start = time_to_seconds(speech['start_time'])
        speech_end = time_to_seconds(speech['end_time'])
        
        # 시간대 겹침 조건 (부분 겹침 포함)
        if (speech_start < vision_end) and (speech_end > vision_start):
            overlapping_speech.append(speech['speech_cap'])

    # 캡션 통합
    merged_caption = ' '.join(overlapping_speech)
    
    # 원본 구조 유지하며 결과 추가
    vision_data[video_id]['merged_speech'] = merged_caption
    
    # 새 파일로 저장
    output_path = vision_caption_path.replace('.json', '_merged.json')
    with open(output_path, 'w') as f:
        json.dump(vision_data, f, indent=2)
    
    return output_path

def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))

    # VideoCaption 인스턴스 생성
    # captioner = VideoCaption(current_dir)
    
    # 비디오 경로 설정
    # media_path = os.path.join(current_dir, 'data', "본인이 원하는 비디오 경로")
    media_path = '/data/ephemeral/home/hanseonglee/level4-cv-finalproject-hackathon-cv-01-lv3/yt8m_Movieclips_xcJXT5lc1Bg_001.mp4'
    # TODO: 오디오 관련 
    input_video_path = '/data/ephemeral/home/hanseonglee/level4-cv-finalproject-hackathon-cv-01-lv3/yt8m_Movieclips_xcJXT5lc1Bg_001.mp4'
    
    input_video_path = '/data/ephemeral/home/hanseonglee/level4-cv-finalproject-hackathon-cv-01-lv3/data/origin/clips'
    video_paths = glob.glob(os.path.join(input_video_path, '*.mp4'))
    # output_audio_path = '/data/ephemeral/home/hanseonglee/level4-cv-finalproject-hackathon-cv-01-lv3/yt8m_Movieclips_xcJXT5lc1Bg_001_audio.wav'
    # 오디오 추출
    # ffmpeg.input(input_video_path).output(output_audio_path, format='wav').run()
    
    # STT 추출
    # stt_result_txt = './stt_result.txt'
    # whisper_model = whisper.load_model("turbo") 
    # for video_path in video_paths:
    #     # audio = whisper.load_audio(video_path)
    #     # mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    #     # detect the spoken language
    #     # _, probs = whisper_model.detect_language(mel)
    #     # print(f"Detected language: {max(probs, key=probs.get)}")
    #     # options = whisper.DecodingOptions(task='translate',language='en')
    #     # result = whisper.decode(whisper_model, mel, options)
    #     # print(result["text"])

    #     audio = whisper.load_audio(video_path)
    #     audio = whisper.pad_or_trim(audio)
    #     print(result["text"])

    #     with open(stt_result_txt, 'a') as f:
    #         # 줄마다 기록
    #         f.write(result["text"] + "\n")
    # print(f"Saved to {stt_result_txt}")
    
if __name__ == "__main__":
    # main()
    # 오디오 파일 경로 설정
    # video_dir = '/data/ephemeral/home/test/YT8M/Movieclips/origin'
    # audio_dir = '/data/ephemeral/home/test/YT8M/Movieclips/origin/audio'
    
    # video_paths = glob.glob(os.path.join(video_dir, '*.mp4'))

    # model = whisper.load_model("large-v3")

    # # json 파일 저장 경로
    # result_dir = video_dir + '/stt'
    # os.makedirs(result_dir, exist_ok=True)

    # # 오디오 추출
    # for video_path in video_paths:
    #     # 오디오 파일 경로 설정
    #     audio_path = os.path.join(audio_dir, os.path.basename(video_path).replace('.mp4', '.wav'))
    #     try:
    #         ffmpeg.input(video_path).output(
    #             audio_path, 
    #             format='wav',
    #             y=True  # FFmpeg CLI의 -y 플래그와 동일
    #         ).run()  
    #     except Exception as e:
    #         print(f"오디오 추출 에러: {e}")
    #     options = dict(task='transcribe', language='en')
    #     result = model.transcribe(audio_path, verbose=True, **options)
    #     print(f"result: {result}")
    #     print(result["text"])
        
    #     # JSON 형식으로 결과 구조화
    #     output_data = []
    #     # HH:MM:SS 형식으로 변환
    #     for segment in result['segments']:
    #         output_data.append({
    #             "start_time": sec_to_time(int(segment['start'])),
    #             "end_time": sec_to_time(int(segment['end'])),
    #             "speech_cap": segment['text'].strip()  
    #         })
        
    #     # JSON 파일 저장
    #     result_json = os.path.join(result_dir, os.path.basename(video_path).replace('.mp4', '.json'))
    #     with open(result_json, 'w') as f:
    #         json.dump(output_data, f, indent=2)
        
    #     print(f"STT 결과가 {result_json} 파일에 저장되었습니다.")







