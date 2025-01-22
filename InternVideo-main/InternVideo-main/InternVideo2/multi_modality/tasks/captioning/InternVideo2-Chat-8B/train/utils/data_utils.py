import cv2
import numpy as np
import torch
import io
import pandas as pd
import os
from torchvision import transforms
import albumentations as A

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from av_utils import load_full_audio_av

# 01.18, deamin
def read_frames_cv2(
    video_path: str = None,
    use_segment: bool = True,
    start_time: int = 0,
    end_time: int = 0,
    s3_client: bool = False,
    # fps: float = 1, # 현재는 동영상 fps를 기준으로 샘플링, 추후 fps를 변경할 때 추가, 
    # sampling: str = 'static', # static, dynamic, dynamic algorithm 토의 후 추가
):
    '''
    단일 segment 대해 frame을 제공하는 함수, 전체 video는 추가 중
    Video를 받아서 프레임을 tensor로 반환한다.
    
    Args: 
        video_path: str, 동영상 경로
        use_segment: bool, 단일 segment 사용 여부
        start_time: int, 시작 시간
        end_time: int, 종료 시간
        s3_client: bool, s3 클라이언트 사용 여부
    
    Returns:
        frames: torch.Tensor, 동영상 프레임 텐서, (T, C, H, W), torch.uint8
    '''
    
    # 동영상 읽기 실패 시 예외 처리 포함, s3와 local 경로 제공
    try:
        if not s3_client:
            video = cv2.VideoCapture(video_path)
            video_fps: float = video.get(cv2.CAP_PROP_FPS)
        else:
            video = cv2.VideoCapture(io.BytesIO(s3_client.get(video_path)))
            video_fps: float = video.get(cv2.CAP_PROP_FPS)
    except Exception as e:
        raise Exception(f"Failed to read video, error: {e}")
    
    frames = []
    if not video.isOpened() or video.get(cv2.CAP_PROP_FRAME_COUNT) == 0:
        raise Exception(f"Failed to open video: {video_path}")
    else:
        #print(f"Successfully opened video: {video_path}")
        pass
    
    # 단일 segment -> else (01.18, deamin)
    if not use_segment:
        frame_indices: list[int, int] = [start_time * video_fps, end_time * video_fps] # [start_time, end_time]
    else:
        # 전체 frame 사용
        frame_indices: list[int, int] = [0, int(video.get(cv2.CAP_PROP_FRAME_COUNT))]

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
    #print(f"frame_indices: {frame_indices}")
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    #print(f"total_frames: {total_frames}, fps: {fps}")

    # segment에 해당하는 frame만 추가
    for idx in range(frame_indices[0], 10):    
        ret, frame = video.read()
        if not ret:
            raise Exception(f"Failed to read frame: {idx}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        # print(f"success processing frame idx: {idx} frames")
    video.release()
    
    # (T, H, W, C) to (T, C, H, W), numpy to tensor, dtype=torch.uint8
    frames = torch.tensor(np.stack(frames), dtype=torch.uint8).permute(0, 3, 1, 2)
    return frames, frame_indices, int(total_frames/fps)

# 01.18, deamin
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A

class InternVideo2_VideoChat2_Dataset(Dataset): 
    '''
    InternVideo2_VideoChat2_Dataset 클래스
    CSV로부터 segment를 불러오고, frame 변환 및 preprocess를 진행하여 Tensor로 반환한다.
    
    Args:
        csv_path: str, CSV 파일 경로
        video_root: str, 동영상 루트 경로(segments 폴더 경로)
        use_segment: bool, 단일 segment 사용 여부
        start_time: int, 시작 시간
        end_time: int, 종료 시간
        s3_client: bool, s3 클라이언트 사용 여부
        use_audio: bool, 오디오 사용 여부
        audio_reader_type: str, 오디오 로더 타입
        train: bool, 학습 여부
    
    Returns:
        frames: pre-processed frames of a one segment, (T, C, H, W), torch.uint8
        audio: (optional)pre-processed audio-spectrogram of a one segment, (T, 1), torch.float32
        annotation: video caption, str
        frame_indices: frame indices, list[int, int]
        segment_name: segment name, str
        duration: segment duration, float
        video_path: video path, str
        annotation: video caption, str
    '''
    def __init__(
            self,
            csv_path: str = None,
            video_root: str = None,
            use_segment: bool = True,
            # start_time: int = 0, 현재는 segment_name을 파싱하여 대체 중
            # end_time: int = 0,
            s3_client: bool = False,
            use_audio: bool = False,
            audio_reader_type: str = 'lazy_load_audio',
            audio_sample_rate: int = 16000,
            max_audio_length: int = 1000000,
            train: bool = True,
    ):
        # video
        assert csv_path is not None and isinstance(csv_path, str), "csv_path must be a string, or not None"
        
        # csv 파일 읽기
        # json 파일 읽어올 수 있도록 수정
        self.segments_df: pd.DataFrame = pd.read_csv(csv_path)
        self.video_root: str = video_root
        self.use_segment: bool = use_segment
        self.s3_client: bool = s3_client
        self.train: bool = train
        self.use_audio: bool = use_audio
        self.audio_reader_type: str = audio_reader_type
        self.audio_sample_rate: int = audio_sample_rate
        self.max_audio_length: int = max_audio_length
        
    def __len__(self):
        # 총 sample 수(segment 수) 반환, train/test 여부에 따라 다르게 반환
        if self.train:
            return len(self.segments_df[self.segments_df['TRAIN/TEST'] == 'train'])
        else:
            return len(self.segments_df[self.segments_df['TRAIN/TEST'] == 'test'])
    
    def __getitem__(self, index):
        if self.train:
            segment_df = self.segments_df[self.segments_df['TRAIN/TEST'] == 'train']
        else:
            segment_df = self.segments_df[self.segments_df['TRAIN/TEST'] == 'test']
            
        # segment 예시: "'ViDEOPATH'_'STARTTIME(HH_MM_SS)'_'ENDTIME(HH_MM_SS)'"
        segment_name = segment_df.iloc[index]['SEGMENT_NAME']
        #print(f"segment_name: {segment_name}")
        parts = segment_name.split('_')
        #print(f"parts: {parts}")
        video_name = parts[0]
        #print(f"video_name: {video_name}")
        start_time = '_'.join(parts[1:4])
        #print(f"start_time: {start_time}")
        end_time = '_'.join(parts[4:7])
        #print(f"end_time: {end_time}")

        
        annotation = segment_df.iloc[index]['ANNO']
        #print(f"annotation: {annotation}")
        video_path = os.path.join(self.video_root, video_name)
        assert video_path is not None and isinstance(video_path, str), "video_path must be a string, or not None"
        assert annotation is not None and isinstance(annotation, str), "annotation must be a string, or not None"
        
        # HH_MM_SS 형식을 초 단위로 변환
        h, m, s = map(int, start_time.split('_'))
        start_time = h * 3600 + m * 60 + s
        
        h, m, s = map(int, end_time.split('_'))
        end_time = h * 3600 + m * 60 + s
        
        # 동영상 읽기
        frames, frame_indices, duration = read_frames_cv2(
            video_path, self.use_segment, start_time, end_time, self.s3_client)
        
        # audio, segment이므로 full audio 사용
        if self.use_audio:
            audio = load_full_audio_av(
                video_path, frame_indices, self.audio_reader_type, self.audio_sample_rate, self.max_audio_length, self.s3_client)
        else:
            audio = None

        
        data = {
            'frames': self.preprocess_frames(frames),
            'audio': audio,
            'frame_indices': frame_indices,
            'segment_name': video_name + '_' + str(start_time) + '_' + str(end_time),
            'duration': duration,
            'video_path': video_path,
            'annotation': annotation
        }
        return data
    
        # index를 통해서 Dataset에 접근했을 때,
        # 해당 Index에 해당하는 segment의 정보들을 반환할 수 있도록만 유지해주시면 됩니다.
    
    def preprocess_frames(self, frames, use_albumentations: bool = False):
        '''
        각 프레임을 전처리하는 함수,
        common transform 적용, albumentations 사용 시 추가 적용 가능
        '''
        # common transform 적용
        if not hasattr(self, 'transform'):
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])

        try:
            # frames를 float32로 변환하고 정규화 (0-1 범위)
            frames = frames.float() / 255.0
            
            # (T, C, H, W) 형태로, 각 프레임에 대한 전처리 진행
            # 전체 프레임에 대해 한 번에 resize 적용
            T, C, H, W = frames.shape
            frames = frames.contiguous().view(-1, C, H, W)  # (T * C, H, W)
            #print(f"frames shape: {frames.shape}")
            frames = self.transform(frames)
            frames = frames.view(T, C, 224, 224)  # (T, C, 224, 224)
  
            #print(f"Input frames shape: {frames.shape}")
            #print(f"Input frames dtype: {frames.dtype}")
            return frames
        
        except Exception as e:
            raise RuntimeError(f"Error processing frames: {str(e)}")
    
    # 데이터셋 정보 출력
    def __repr__(self):
        return f"""InternVideo2_VideoChat2_Dataset(
            num_frames: {len(self.frame_indices)}
            duration: {self.duration:.2f}s
            frame_shape: {self.frames[0].shape if len(self.frames) > 0 else None}
        )"""
    
# 01.18, deamin
from torch.utils.data import DataLoader
class InternVideo2_VideoChat2_DataLoader(DataLoader):
    def __init__(
        self,
        dataset: InternVideo2_VideoChat2_Dataset,
        batch_size: int = 2,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_audio: bool = False,
    ):
        """
        비디오 데이터 로더 초기화
        Args:
            dataset: InternVideo2_VideoChat2_Dataset, 데이터셋 인스턴스
            batch_size: 배치 크기, 기본 2
            shuffle: 데이터 셔플 여부, 기본 False
            num_workers: 데이터 로딩에 사용할 워커 수, 기본 4
            pin_memory: GPU 메모리 고정 여부, 기본 True
            use_audio: 오디오 사용 여부, 기본 False
        Returns:
            dataloader: torch.utils.data.DataLoader, 데이터로더 인스턴스
        """
        self.use_audio = use_audio
        self.dataset = dataset
        self.batch_size = batch_size
        
        # 오디오 포함 여부에 따른 collate_fn,
        # Dataset에서 반환하는 data: dict 에서 학습에 필요한 데이터 반환 
        def collate_fn(batch):
            if not self.use_audio:
                return {
                    'frames': torch.stack([item['frames'] for item in batch]),
                    'segment_names': [item['segment_name'] for item in batch],
                    'annotations': [item['annotation'] for item in batch] if 'annotation' in batch[0] else None
                }
            else:
                return {
                    'frames': torch.stack([item['frames'] for item in batch]),
                    'audio': torch.stack([item['audio'] for item in batch]),
                    'segment_names': [item['segment_name'] for item in batch],
                    'annotations': [item['annotation'] for item in batch] if 'annotation' in batch[0] else None
                }
        
        # DataLoader 초기화
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
    
    def __iter__(self):
        """데이터로더 이터레이터 반환"""
        return iter(self.dataloader)
    
    def __len__(self):
        """데이터셋의 총 배치 수 반환"""
        return len(self.dataloader)
    
    def get_batch_size(self):
        """현재 배치 크기 반환"""
        return self.batch_size
    
    def get_dataset(self):
        """현재 데이터셋 반환"""
        return self.dataset
    
    def get_use_audio(self):
        """오디오 사용 여부 반환"""
        return self.use_audio