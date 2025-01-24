import cv2
import numpy as np
import torch
import io
import pandas as pd
import os
from torchvision import transforms
import albumentations as A
import json

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .av_utils import load_full_audio_av

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
        print(f"Successfully opened video: {video_path}")
    
    # 단일 segment -> else (01.18, deamin)
    if not use_segment:
        frame_indices: list[int, int] = [start_time * video_fps, end_time * video_fps] # [start_time, end_time]
    else:
        # 전체 frame 사용
        frame_indices: list[int, int] = [0, int(video.get(cv2.CAP_PROP_FRAME_COUNT))]

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
    print(f"frame_indices: {frame_indices}")
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"total_frames: {total_frames}, fps: {fps}")

    # segment에 해당하는 frame만 추가
    for idx in range(frame_indices[0], 8):#frame_indices[1]):    
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
        ### 대체로 인한 삭제 예정
        json_path: str, JSON 파일 경로
        video_root: str, 동영상 루트 경로(segments 폴더 경로)
        ### 대체로 인한 추가 예정
        data_path: str, 데이터 파일 경로
        ###
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
        'start_time' : start_time, str (hh:mm:ss)
        'end_time' : end_time, str (hh:mm:ss)
    '''
    def __init__(
            self,
            data_path: str = "../../data",
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
        assert data_path is not None and isinstance(data_path, str), "data_path must be a string, or not None"
        
        self.labels: list = self.load_label(data_path)
        self.use_segment: bool = use_segment
        self.s3_client: bool = s3_client
        self.train: bool = train
        self.use_audio: bool = use_audio
        self.audio_reader_type: str = audio_reader_type
        self.audio_sample_rate: int = audio_sample_rate
        self.max_audio_length: int = max_audio_length
        
    def __len__(self):
        # json형태의 annotations가 각 segment마다 하나씩 존재하므로, json파일의 개수를 반환하면 됨
        return len(self.labels)
    
    def __getitem__(self, index):
        # segment 예시: "'ViDEOPATH'_'SEGMENTINDEX'"
        with open(self.labels[index], 'r', encoding='utf-8') as file:
            data = json.load(file)
        segment_name = list(data.keys())[0]
        start_time = data[segment_name]['start_time']
        end_time = data[segment_name]['end_time']

        
        annotation = data[segment_name]['caption']
        video_path = self.label_to_video(self.labels[index])
        assert video_path is not None and isinstance(video_path, str), "video_path must be a string, or not None"
        assert annotation is not None and isinstance(annotation, str), "annotation must be a string, or not None"
        
        # HH_MM_SS 형식을 초 단위로 변환
        h, m, s = map(int, start_time.split(':'))
        start_time = h * 3600 + m * 60 + s
        
        h, m, s = map(int, end_time.split(':'))
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
            'segment_name': segment_name,
            'duration': duration,
            'video_path': video_path,
            'annotation': annotation,
            'start_time' : start_time,
            'end_time' : end_time
        }
        return data
    
    def label_to_video(self, label: str) -> str:
        '''
        label의 경로에 따라 그에 맞는 video의 경로를 반환
        '''
        replacements = {'json':'mp4', 'labels':'clips'}
        video = label # To be transformed
        for old, new in replacements.items():
            video = video.replace(old, new)
        return video

    def load_label(self, data_path: str) -> list:
        '''
        data_path 내부에 있는 모든 json형태의 label을 반환
        '''
        
        all_labels = []
        all_clips = []

        ## dsrc는 데이터 출처를 의미 (예: YT8M, MVAD 등)
        for dsrc in os.listdir(data_path): 
            dsrc_path = os.path.join(data_path, dsrc)

            ## category는 데이터의 category를 의미함 (예: movieclips, trailer 등)
            for category in os.listdir(dsrc_path):
                category_path = os.path.join(dsrc_path, category)

                ## directory는 반드시 clips 혹은 labels로만 나누어짐
                for directory in os.listdir(category_path):
                    directory_path = os.path.join(category_path, directory)
                        
                    if directory == 'labels':
                        sub_labels = [os.path.join(directory_path, x) for x in os.listdir(directory_path) if x.endswith('json')]
                        all_labels.extend(sub_labels)

                    elif directory == 'clips':
                        sub_clips = [os.path.join(directory_path, x) for x in os.listdir(directory_path) if x.endswith('mp4')]
                        all_clips.extend(sub_clips)
        self.integrity_check(all_labels, all_clips)
        return all_labels

    def integrity_check(self, labels, clips):
        '''
        모든 labels가 clips와 매칭되는 지 확인합니다
        '''
        transformed_labels = [self.label_to_video(x) for x in labels]
        mismatch_no_clips = set(transformed_labels) - set(clips)
        mismatch_no_labels = set(clips) - set(transformed_labels)

        if len(labels) == 0 and len(clips) == 0:
            raise RuntimeError(f"No datas found")
        if len(labels) == 0:
            raise RuntimeError(f"No labels found")
        if len(clips) == 0:
            raise RuntimeError(f"No clips found")
            
        if len(mismatch_no_clips) > 0 or len(mismatch_no_labels) > 0:
            mismatched_samples_clips = [os.path.basename(x) for x in mismatch_no_clips]
            mismatched_samples_labels = [os.path.basename(x).replace('mp4', 'json') for x in mismatch_no_labels]
            raise RuntimeError(f"{len(mismatch_no_clips) + len(mismatch_no_labels)} sample(s) mismatched!\n Missing clips: {mismatched_samples_clips}\n Missing labels: {mismatched_samples_labels}")

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
            print(f"frames shape: {frames.shape}")
            frames = self.transform(frames)
            frames = frames.view(T, C, 224, 224)  # (T, C, 224, 224)

            print(f"Input frames shape: {frames.shape}")
            print(f"Input frames dtype: {frames.dtype}")
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
                    'start_times' : [item['start_time'] for item in batch],
                    'end_times' : [item['end_time'] for item in batch],
                    'annotations': [item['annotation'] for item in batch] if 'annotation' in batch[0] else None,
                }
            else:
                return {
                    'frames': torch.stack([item['frames'] for item in batch]),
                    'audio': torch.stack([item['audio'] for item in batch]),
                    'segment_names': [item['segment_name'] for item in batch],
                    'start_times' : [item['start_time'] for item in batch],
                    'end_times' : [item['end_time'] for item in batch],
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