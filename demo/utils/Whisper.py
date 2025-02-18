import os
import json
import torch
import whisper
import ffmpeg
class Whisper:
    """
    Whisper 모델을 사용하여 오디오를 텍스트로 변환하는 클래스

    example:
        >> whisper = Whisper()
        >> whisper.load_video("video_path")
        >> text = whisper.audio2text()
        >> return text >> "speech to text information"
    """
    def __init__(self, audio_path="./audio", model_type="turbo"):
        self.model = whisper.load_model(model_type)
        self.model.eval()
        self.json_dir = "./video"
        self.audio_path = audio_path
        os.makedirs(self.json_dir, exist_ok=True)  # 디렉토리 생성 추가

    def speech_to_text(self, video_path):
        """
        STT를 수행하고 json 파일로 저장
        """ 
        result_audio_path = os.path.join(self.audio_path, os.path.basename(video_path).replace(".mp4", ".wav"))
        try:
            ffmpeg.input(video_path).output(  
                result_audio_path, 
                format='wav',
                y=True  # FFmpeg CLI의 -y 플래그와 동일
            ).run()  
        except Exception as e:
            print(f"오디오 추출 에러: {e}")
            return None
        options = dict(task='transcribe', language='en')
        result = self.model.transcribe(result_audio_path, verbose=True, **options)
        
        output_data = []
        # HH:MM:SS 형식으로 변환
        for segment in result['segments']:
            output_data.append({
                "start_time": self._sec_to_time(int(segment['start'])),
                "end_time": self._sec_to_time(int(segment['end'])),
                "speech_cap": segment['text'].strip()  
            })
        
        # # JSON 파일 저장
        # self.result_json = os.path.join(self.json_dir, os.path.basename(self.video_path).replace('.mp4', '.json'))
        # with open(self.result_json, 'w') as f:
        #     json.dump(output_data, f, indent=2)
        return output_data



    # def _time_to_seconds(self,time_str):
    #     """HH:MM:SS 형식을 초 단위로 변환"""
    #     h, m, s = map(int, time_str.split(':'))
    #     return h * 3600 + m * 60 + s

    def _sec_to_time(self,sec: int) -> str:
        """
        초(sec)을 시:분:초로 변환할 수 있는 함수입니다.
        sec: 특정 시점의 초(sec)
        """
        s = sec % 60
        m = sec // 60
        h = sec // 3600
        return f"{h:02d}:{m:02d}:{s:02d}"
    # def merge_captions_from_json(self, vision_caption_path, use_segment=False):
        # if use_segment:
        #     # 세그먼트 비디오 파일명 기반으로 JSON 파일 찾기
        #     segment_base = os.path.basename(vision_caption_path).replace('.mp4', '')
        #     vision_json = os.path.join(self.json_dir, f"{segment_base}.json")
            
        #     with open(vision_json, 'r') as f:
        #         vision_data = json.load(f)
        #     with open(self.result_json, 'r') as f:
        #         speech_caption = json.load(f)

        #     # 비전 타임라인 추출
        #     video_id = next(iter(vision_data))  # 첫 번째 키 추출 (e.g. "yt8m_Movieclips_xcJXT5lc1Bg_001")
        #     vision_start = self._time_to_seconds(vision_data[video_id]['start_time'])
        #     vision_end = self._time_to_seconds(vision_data[video_id]['end_time'])
            
        #     # 시간대 필터링
        #     overlapping_speech = []
        #     for speech in speech_caption:
        #         speech_start = self._time_to_seconds(speech['start_time'])
        #         speech_end = self._time_to_seconds(speech['end_time'])
        #         # 시간대 겹침 조건 (부분 겹침 포함)
        #         if (speech_start < vision_end) and (speech_end > vision_start):
        #             overlapping_speech.append(speech['speech_cap'])

        #     # 캡션 통합
        #     merged_caption = ' '.join(overlapping_speech)
        #     return merged_caption
        # else:
        #     with open(self.result_json, 'r') as f:
        #         speech_caption = json.load(f)
        #     return speech_caption
        


