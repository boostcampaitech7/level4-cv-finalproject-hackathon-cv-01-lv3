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
import numpy as np
from sampling import read_frames_cv2, read_frame_cv2
# 01.18, deamin
from torch.utils.data import Dataset
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InternVideo2_VideoChat2_Dataset(Dataset): 
    '''
    InternVideo2_VideoChat2_Dataset 클래스
    segment를 불러오고, frame 변환 및 preprocess를 진행하여 Tensor로 반환한다.
    
    Args:
        data_path: str, 데이터 파일 경로
        use_segment: bool, 단일 segment 사용 여부
        start_time: int, 시작 시간
        end_time: int, 종료 시간
        s3_client: bool, s3 클라이언트 사용 여부
        use_audio: bool, 오디오 사용 여부
        audio_reader_type: str, 오디오 로더 타입
        train: bool, 학습 여부
        num_frames: int, 등간격으로 몇 개의 프레임을 추출하여 모델에 넣어줄 지 결정
        save_frames_as_img: bool, 샘플링되는 프레임 저장 여부
        resize: int, 전처리로 재조정할 사이즈 (Default: 224)
    
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
            num_frames: int = 16,
            save_frames_as_img: bool = False,
            resize: int = 224,
    ):
        # video
        assert data_path is not None and isinstance(data_path, str), "data_path must be a string, or not None"
        
        self.labels: list = self.load_label(data_path, train='train' if train else 'test')
        self.use_segment: bool = use_segment
        self.s3_client: bool = s3_client
        self.train: bool = train
        self.use_audio: bool = use_audio
        self.audio_reader_type: str = audio_reader_type
        self.audio_sample_rate: int = audio_sample_rate
        self.max_audio_length: int = max_audio_length
        self.num_frames: int = num_frames
        self.save_frames_as_img: bool = save_frames_as_img
        self.resize: int = resize
        
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
            video_path, self.use_segment, start_time, end_time, self.s3_client, num_frames = self.num_frames, save_frames_as_img = self.save_frames_as_img)
        
        # audio, segment이므로 full audio 사용
        if self.use_audio:
            audio = load_full_audio_av(
                video_path, frame_indices, self.audio_reader_type, self.audio_sample_rate, self.max_audio_length, self.s3_client)
        else:
            audio = None

        
        data = {
            'frames': self.preprocess_frames(frames, resize=self.resize),
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

    def load_label(self, data_path: str, train: str) -> list:
        '''
        data_path 내부에 있는 모든 json형태의 label을 반환
        '''
        
        all_labels = []
        all_clips = []

        ## dsrc는 데이터 출처를 의미 (예: YT8M, MVAD 등)
        if not os.path.isdir(data_path):
            raise NotADirectoryError(f'{data_path} is not a directory! Please set data_path as proper directory path \n Inside data_path, there should be data/your_data_source/your_category!')
        for dsrc in sorted(os.listdir(data_path)):
            dsrc_path = os.path.join(data_path, dsrc)
            if not os.path.isdir(dsrc_path):
                continue

            ## category는 데이터의 category를 의미함 (예: movieclips, trailer 등)
            for category in sorted(os.listdir(dsrc_path)):
                category_path = os.path.join(dsrc_path, category)
                if not os.path.isdir(category_path):
                    continue

                ## directory는 반드시 먼저 train 또는 test로 이루어지며, 이후, clips 혹은 labels로 나누어짐
                for directory in sorted(os.listdir(os.path.join(category_path, train))):
                    directory_path = os.path.join(category_path, train, directory)
                    if not os.path.isdir(directory_path):
                        continue
                    # train인지 test인지. task와 동일하다면 진행 아니면 pass
                    if directory == 'labels':
                        sub_labels = [os.path.join(directory_path, x) for x in sorted(os.listdir(directory_path)) if x.endswith('json')]
                        all_labels.extend(sub_labels)

                    elif directory == 'clips':
                        sub_clips = [os.path.join(directory_path, x) for x in sorted(os.listdir(directory_path)) if x.endswith('mp4')]
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

    def preprocess_frames(self, frames, use_albumentations: bool = False, resize: int = 224):
        '''
        각 프레임을 전처리하는 함수,
        common transform 적용, albumentations 사용 시 추가 적용 가능
        '''
        # common transform 적용
        if not hasattr(self, 'transform'):
            self.transform = transforms.Compose([
                transforms.Resize((resize, resize)),
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
            # print(f"frames shape: {frames.shape}")
            frames = self.transform(frames)
            frames = frames.view(T, C, resize, resize)  # (T, C, 224, 224)

            # print(f"Input frames shape: {frames.shape}")
            # print(f"Input frames dtype: {frames.dtype}")
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
                    'video_paths': [item['video_path'] for item in batch]
                }
            else:
                return {
                    'frames': torch.stack([item['frames'] for item in batch]),
                    'audio': torch.stack([item['audio'] for item in batch]),
                    'segment_names': [item['segment_name'] for item in batch],
                    'start_times' : [item['start_time'] for item in batch],
                    'end_times' : [item['end_time'] for item in batch],
                    'annotations': [item['annotation'] for item in batch] if 'annotation' in batch[0] else None,
                    'video_paths': [item['video_path'] for item in batch]
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
    

class InternVideo2_VideoChat2_Image_Dataset(Dataset): 
    '''
    InternVideo2_VideoChat2_Image_Dataset 클래스
    segment를 불러오고, frame 변환 및 preprocess를 진행하여 Tensor로 반환한다.
    
    Args:
        data_path: str, 데이터 파일 경로
        use_segment: bool, 단일 segment 사용 여부
        start_time: int, 시작 시간
        end_time: int, 종료 시간
        s3_client: bool, s3 클라이언트 사용 여부
        use_audio: bool, 오디오 사용 여부
        audio_reader_type: str, 오디오 로더 타입
        train: bool, 학습 여부
        num_frames: int, 등간격으로 몇 개의 프레임을 추출하여 모델에 넣어줄 지 결정
        offset: int, 기준 프레임으로부터 +- 몇 개의 프레임(오프셋)을 사용할 지 결정
        save_frames_as_img: bool, 샘플링되는 프레임 저장 여부
    
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
            s3_client: bool = False,
            use_audio: bool = False,
            audio_reader_type: str = 'lazy_load_audio',
            audio_sample_rate: int = 16000,
            max_audio_length: int = 1000000,
            num_frames: int = 2,
            train: bool = True,
            save_frames_as_img: bool = False,
    ):
        # image
        assert data_path is not None and isinstance(data_path, str), "data_path must be a string, or not None"
        
        self.labels: list = self.load_label(data_path, train='train' if train else 'test')
        self.use_segment: bool = use_segment
        self.s3_client: bool = s3_client
        self.train: bool = train
        # self.use_audio: bool = use_audio
        # self.audio_reader_type: str = audio_reader_type
        # self.audio_sample_rate: int = audio_sample_rate
        # self.max_audio_length: int = max_audio_length
        self.save_frames_as_img: bool = save_frames_as_img
        self.num_frames = num_frames
        self.all_frames = self._prepare_frames()
    
    def _prepare_frames(self):
        # segment 예시: "'ViDEOPATH'_'SEGMENTINDEX'"
        all_frames = []
        for label in self.labels:
            with open(label, 'r', encoding='utf-8') as file:
                data = json.load(file)
            segment_name = list(data.keys())[0]
            start_time = data[segment_name]['start_time']
            end_time = data[segment_name]['end_time']

            
            annotation = data[segment_name]['caption']
            video_path = self.label_to_video(label)
            assert video_path is not None and isinstance(video_path, str), "video_path must be a string, or not None"
            assert annotation is not None and isinstance(annotation, str), "annotation must be a string, or not None"
            
            # HH_MM_SS 형식을 초 단위로 변환
            h, m, s = map(int, start_time.split(':'))
            start_time = h * 3600 + m * 60 + s
            
            h, m, s = map(int, end_time.split(':'))
            end_time = h * 3600 + m * 60 + s
            
            # 동영상 읽기
            frames, frame_indices = read_frame_cv2(
            video_path, self.use_segment, start_time, end_time, self.s3_client, num_frames = self.num_frames, save_frames_as_img = self.save_frames_as_img)

            for frame, frame_index in zip(frames, frame_indices):
                all_frames.append({
                'frame': self.preprocess_frames(frame),
                'frame_index': frame_index,
                'segment_name': segment_name,
                'annotation': annotation,
            })
        return all_frames
    
    def __getitem__(self, index):
        return self.all_frames[index]
    
    def __len__(self):
        return len(self.all_frames)
    
    def label_to_video(self, label: str) -> str:
        '''
        label의 경로에 따라 그에 맞는 video의 경로를 반환
        '''
        replacements = {'json':'mp4', 'labels':'clips'}
        video = label # To be transformed
        for old, new in replacements.items():
            video = video.replace(old, new)
        return video

    def load_label(self, data_path: str, train: str) -> list:
        '''
        data_path 내부에 있는 모든 json형태의 label을 반환
        '''
        
        all_labels = []
        all_clips = []

        ## dsrc는 데이터 출처를 의미 (예: YT8M, MVAD 등)
        if not os.path.isdir(data_path):
            raise NotADirectoryError(f'{data_path} is not a directory! Please set data_path as proper directory path \n Inside data_path, there should be data/your_data_source/your_category!')
        for dsrc in sorted(os.listdir(data_path)):
            dsrc_path = os.path.join(data_path, dsrc)
            if not os.path.isdir(dsrc_path):
                continue

            ## category는 데이터의 category를 의미함 (예: movieclips, trailer 등)
            for category in sorted(os.listdir(dsrc_path)):
                category_path = os.path.join(dsrc_path, category)
                if not os.path.isdir(category_path):
                    continue

                ## directory는 반드시 먼저 train 또는 test로 이루어지며, 이후, clips 혹은 labels로 나누어짐
                for directory in sorted(os.listdir(os.path.join(category_path, train))):
                    directory_path = os.path.join(category_path, train, directory)
                    if not os.path.isdir(directory_path):
                        continue
                    # train인지 test인지. task와 동일하다면 진행 아니면 pass
                    if directory == 'labels':
                        sub_labels = [os.path.join(directory_path, x) for x in sorted(os.listdir(directory_path)) if x.endswith('json')]
                        all_labels.extend(sub_labels)

                    elif directory == 'clips':
                        sub_clips = [os.path.join(directory_path, x) for x in sorted(os.listdir(directory_path)) if x.endswith('mp4')]
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

    def preprocess_frames(self, frame, use_albumentations: bool = False):
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
            frame = frame.float() / 255.0
            
            # (T, C, H, W) 형태로, 각 프레임에 대한 전처리 진행
            # 전체 프레임에 대해 한 번에 resize 적용
            C, H, W = frame.shape
            frame = frame.contiguous().view(C, H, W)  # (C, H, W)
            # print(f"frames shape: {frames.shape}")
            frame = self.transform(frame)
            print(f"frame.max: {frame.max()}, frame.min: {frame.min()}")
            frame = frame.view(C, 224, 224)  # (C, 224, 224)

            # print(f"Input frames shape: {frames.shape}")
            # print(f"Input frames dtype: {frames.dtype}")
            return frame
        
        except Exception as e:
            raise RuntimeError(f"Error processing frames: {str(e)}")
    
    # 데이터셋 정보 출력
    def __repr__(self):
        return f"""InternVideo2_VideoChat2_Image_Dataset(
            num_frames: {len(self.all_frames)}
            frame_shape: {self.all_frames[0].shape if len(self.all_frames) > 0 else None}
        )"""
    

class InternVideo2_VideoChat2_Image_DataLoader(DataLoader):
    def __init__(
        self,
        dataset: InternVideo2_VideoChat2_Image_Dataset,
        batch_size: int = 2,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        이미지 데이터 로더 초기화
        Args:
            dataset: InternVideo2_VideoChat2_Image_Dataset, 데이터셋 인스턴스
            batch_size: 배치 크기, 기본 2
            shuffle: 데이터 셔플 여부, 기본 False
            num_workers: 데이터 로딩에 사용할 워커 수, 기본 4
            pin_memory: GPU 메모리 고정 여부, 기본 True
        Returns:
            dataloader: torch.utils.data.DataLoader, 데이터로더 인스턴스
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Dataset에서 반환하는 data: dict 에서 학습에 필요한 데이터 반환 
        def collate_fn(batch):
            return {
                'frames': torch.stack([item['frame'] for item in batch]),
                'segment_names': [item['segment_name'] for item in batch],
                'frame_indices' : [item['frame_index'] for item in batch],
                'annotations': [item['annotation'] for item in batch] if 'annotation' in batch[0] else None,
            }
        
        # DataLoader 초기화
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,  # <- 여기에 추가
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

class BLIP2_Image_Dataset(Dataset): 
    '''
    BLIP2_Image_Dataset 클래스
    segment를 불러오고, frame 변환 및 preprocess를 진행하여 Tensor로 반환한다.
    
    Args:
        data_path: str, 데이터 파일 경로
        use_segment: bool, 단일 segment 사용 여부
        start_time: int, 시작 시간
        end_time: int, 종료 시간
        s3_client: bool, s3 클라이언트 사용 여부
        use_audio: bool, 오디오 사용 여부
        audio_reader_type: str, 오디오 로더 타입
        train: bool, 학습 여부
        num_frames: int, 등간격으로 몇 개의 프레임을 추출하여 모델에 넣어줄 지 결정
        offset: int, 기준 프레임으로부터 +- 몇 개의 프레임(오프셋)을 사용할 지 결정
        save_frames_as_img: bool, 샘플링되는 프레임 저장 여부
    
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
            num_frames: int = 2,
            train: bool = True,
            save_frames_as_img: bool = False,
    ):
        # image
        assert data_path is not None and isinstance(data_path, str), "data_path must be a string, or not None"
        
        self.labels: list = self.load_label(data_path, train='train' if train else 'test')
        self.train: bool = train
        self.save_frames_as_img: bool = save_frames_as_img
        self.num_frames = num_frames
        self.all_frames = self._prepare_frames()
    
    def _prepare_frames(self):
        # segment 예시: "'ViDEOPATH'_'SEGMENTINDEX'"
        all_frames = []
        for label in self.labels:
            with open(label, 'r', encoding='utf-8') as file:
                data = json.load(file)
            segment_name = list(data.keys())[0]
            start_time = data[segment_name]['start_time']
            end_time = data[segment_name]['end_time']

            
            annotation = data[segment_name]['caption']
            video_path = self.label_to_video(label)
            assert video_path is not None and isinstance(video_path, str), "video_path must be a string, or not None"
            assert annotation is not None and isinstance(annotation, str), "annotation must be a string, or not None"
            
            # HH_MM_SS 형식을 초 단위로 변환
            h, m, s = map(int, start_time.split(':'))
            start_time = h * 3600 + m * 60 + s
            
            h, m, s = map(int, end_time.split(':'))
            end_time = h * 3600 + m * 60 + s
            
            # 동영상 읽기
            frames, frame_indices = read_frame_cv2(
            video_path, True, start_time, end_time, None, num_frames = self.num_frames, save_frames_as_img = self.save_frames_as_img)

            for frame, frame_index in zip(frames, frame_indices):
                all_frames.append({
                'frame': self.preprocess_frames(frame),
                'frame_index': frame_index,
                'segment_name': segment_name,
                'annotation': annotation,
            })
        return all_frames
    
    def __getitem__(self, index):
        return self.all_frames[index]
    
    def __len__(self):
        return len(self.all_frames)
    
    def label_to_video(self, label: str) -> str:
        '''
        label의 경로에 따라 그에 맞는 video의 경로를 반환
        '''
        replacements = {'json':'mp4', 'labels':'clips'}
        video = label # To be transformed
        for old, new in replacements.items():
            video = video.replace(old, new)
        return video

    def load_label(self, data_path: str, train: str) -> list:
        '''
        data_path 내부에 있는 모든 json형태의 label을 반환
        '''
        
        all_labels = []
        all_clips = []

        ## dsrc는 데이터 출처를 의미 (예: YT8M, MVAD 등)
        if not os.path.isdir(data_path):
            raise NotADirectoryError(f'{data_path} is not a directory! Please set data_path as proper directory path \n Inside data_path, there should be data/your_data_source/your_category!')
        for dsrc in sorted(os.listdir(data_path)):
            dsrc_path = os.path.join(data_path, dsrc)
            if not os.path.isdir(dsrc_path):
                continue

            ## category는 데이터의 category를 의미함 (예: movieclips, trailer 등)
            for category in sorted(os.listdir(dsrc_path)):
                category_path = os.path.join(dsrc_path, category)
                if not os.path.isdir(category_path):
                    continue

                ## directory는 반드시 먼저 train 또는 test로 이루어지며, 이후, clips 혹은 labels로 나누어짐
                for directory in sorted(os.listdir(os.path.join(category_path, train))):
                    directory_path = os.path.join(category_path, train, directory)
                    if not os.path.isdir(directory_path):
                        continue
                    # train인지 test인지. task와 동일하다면 진행 아니면 pass
                    if directory == 'labels':
                        sub_labels = [os.path.join(directory_path, x) for x in sorted(os.listdir(directory_path)) if x.endswith('json')]
                        all_labels.extend(sub_labels)

                    elif directory == 'clips':
                        sub_clips = [os.path.join(directory_path, x) for x in sorted(os.listdir(directory_path)) if x.endswith('mp4')]
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

    def preprocess_frames(self, frame, use_albumentations: bool = False):
        '''
        각 프레임을 전처리하는 함수,
        common transform 적용, albumentations 사용 시 추가 적용 가능
        '''
        try:
            C, H, W = frame.shape
            frame = frame.contiguous().view(C, H, W)  # (C, H, W)
            return frame
        
        except Exception as e:
            raise RuntimeError(f"Error processing frames: {str(e)}")
    
    # 데이터셋 정보 출력
    def __repr__(self):
        return f"""InternVideo2_VideoChat2_Image_Dataset(
            num_frames: {len(self.all_frames)}
            frame_shape: {self.all_frames[0].shape if len(self.all_frames) > 0 else None}
        )"""
    
    
class InternVL_Video_Dataset(Dataset):
    '''
    InternVL2.5_Video_Dataset 클래스
    segment를 불러오고, frame 변환 및 preprocess를 진행하여 Tensor로 반환한다.
    
    Args:
        data_path: str, 데이터 파일 경로
        num_frames: int, 등간격으로 몇 개의 프레임을 추출하여 모델에 넣어줄 지 결정
        save_frames_as_img: bool, 샘플링되는 프레임 저장 여부
        input_size: int, 모델에 입력될 이미지의 크기
        min_num: int, HD 기법으로 이미지를 변형하기 위한 파라미터 (Aspect Ratio가 결정됨)
        max_num: int, HD 기법으로 이미지를 변형하기 위한 파라미터 (Aspect Ratio가 결정됨)
    
    Returns:
        pixel_values: pre-processed frames of a one segment, (T, C, H, W), torch.uint8
        annotation: video caption, str
        segment_name: segment name, str
        start_time: start_time, str (hh:mm:ss)
        end_time: end_time, str (hh:mm:ss)
    '''
    def __init__(self, 
            data_path: str = "../../data",
            num_frames: int = 2,
            train: bool = True,
            save_frames_as_img: bool = False,
            input_size: int = 448,
            min_num: int = 1,
            max_num: int = 1,
            use_thumbnail: bool = True):
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD
        
        self.labels: list = self.load_label(data_path, train='train' if train else 'test')
        self.num_segments = num_frames
        self.input_size = input_size
        self.min_num = min_num
        self.max_num = max_num
        self.use_thumbnail = use_thumbnail
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        with open(self.labels[index], 'r', encoding='utf-8') as file:
            data = json.load(file)
        segment_name = list(data.keys())[0]
        start_time = data[segment_name]['start_time']
        end_time = data[segment_name]['end_time']

        
        annotation = data[segment_name]['caption']
        video_path = self.label_to_video(self.labels[index])
        assert video_path is not None and isinstance(video_path, str), "video_path must be a string, or not None"
        assert annotation is not None and isinstance(annotation, str), "annotation must be a string, or not None"
        
        pixel_values, num_patches_list = self.load_video(video_path, num_segments=self.num_segments, max_num=self.max_num)
        return {'segment_name': segment_name,
                'pixel_values': pixel_values,
                'num_patches_list': num_patches_list,
                'annotation': annotation,
                'start_time' : start_time,
                'end_time' : end_time
                }
    
    def label_to_video(self, label: str) -> str:
        '''
        label의 경로에 따라 그에 맞는 video의 경로를 반환
        '''
        replacements = {'json':'mp4', 'labels':'clips'}
        video = label # To be transformed
        for old, new in replacements.items():
            video = video.replace(old, new)
        return video

    def load_label(self, data_path: str, train: str) -> list:
        '''
        data_path 내부에 있는 모든 json형태의 label을 반환
        '''
        
        all_labels = []
        all_clips = []

        ## dsrc는 데이터 출처를 의미 (예: YT8M, MVAD 등)
        if not os.path.isdir(data_path):
            raise NotADirectoryError(f'{data_path} is not a directory! Please set data_path as proper directory path \n Inside data_path, there should be data/your_data_source/your_category!')
        for dsrc in sorted(os.listdir(data_path)):
            dsrc_path = os.path.join(data_path, dsrc)
            if not os.path.isdir(dsrc_path):
                continue

            ## category는 데이터의 category를 의미함 (예: movieclips, trailer 등)
            for category in sorted(os.listdir(dsrc_path)):
                category_path = os.path.join(dsrc_path, category)
                if not os.path.isdir(category_path):
                    continue

                ## directory는 반드시 먼저 train 또는 test로 이루어지며, 이후, clips 혹은 labels로 나누어짐
                for directory in sorted(os.listdir(os.path.join(category_path, train))):
                    directory_path = os.path.join(category_path, train, directory)
                    if not os.path.isdir(directory_path):
                        continue
                    # train인지 test인지. task와 동일하다면 진행 아니면 pass
                    if directory == 'labels':
                        sub_labels = [os.path.join(directory_path, x) for x in sorted(os.listdir(directory_path)) if x.endswith('json')]
                        all_labels.extend(sub_labels)

                    elif directory == 'clips':
                        sub_clips = [os.path.join(directory_path, x) for x in sorted(os.listdir(directory_path)) if x.endswith('mp4')]
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

    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        '''
        정해진 개수 (self.num_segments)의 프레임을 추출하는 하여 frame_indices를 반환
        '''
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
    
    def build_transform(self, input_size):
        '''
        Transform을 정의
        '''
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        '''
        최대한 원본 비디오에 잘 맞는 Aspect Ratio를 찾아서 반환
        '''
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

    def dynamic_preprocess(self, image, image_size, max_num, use_thumbnail=False):
        '''
        가장 원본 비디오에 잘 맞는 Aspect Ratio로 각 프레임의 크기를 변환
        '''
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(self.min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= self.min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
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

    def load_video(self, video_path, num_segments, max_num, bound=None):
        '''
        비디오를 입력으로 받아 Pixel_values (Frames)와 패치의 개수 (num_segments와 동일)을 반환
        '''
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=self.input_size)
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = self.dynamic_preprocess(img, image_size=self.input_size, use_thumbnail=self.use_thumbnail, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list
    

class InternVL_Video_DataLoader(DataLoader):
    def __init__(
        self,
        dataset: InternVL_Video_Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        비디오 데이터 로더 초기화
        Args:
            dataset: InternVL_Video_Dataset, 데이터셋 인스턴스
            batch_size: 배치 크기, 기본 2
            shuffle: 데이터 셔플 여부, 기본 False
            num_workers: 데이터 로딩에 사용할 워커 수, 기본 4
            pin_memory: GPU 메모리 고정 여부, 기본 True
        Returns:
            dataloader: torch.utils.data.DataLoader, 데이터로더 인스턴스
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        def collate_fn(batch):
            return {
                    'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
                    'segment_names': [item['segment_name'] for item in batch],
                    'start_times' : [item['start_time'] for item in batch],
                    'end_times' : [item['end_time'] for item in batch],
                    'annotations': [item['annotation'] for item in batch] if 'annotation' in batch[0] else None,
                    'num_patches_lists': [item['num_patches_list'] for item in batch]
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