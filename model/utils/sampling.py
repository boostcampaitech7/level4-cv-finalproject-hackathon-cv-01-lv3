import cv2
import numpy as np
import torch
import os
from decord import VideoReader, cpu
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms as T

def extract_keyframes(video_path: str, num_frames:int=8) -> list:
    '''
    # 현재 쓰이지 않는 기능
    OpticalFlow를 기반으로 프레임 간의 차이를 계산하여 샘플링할 프레임 번호를 반환하는 함수입니다.
    Args:
        video_path (str): 동영상 경로
        num_frames (int): 반환할 프레임 수

    Returns:
        list: 선택된 프레임 번호 리스트
    '''
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_diffs = []

    # 모든 프레임의 차이 계산
    prev_frame = None
    for i in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break
        # 프레임 간 픽셀 값이 얼마나 유사한지 파악하기 위해 grayscale로 변환 후, 차이 계산
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.asarray(gray, dtype=np.uint8)
        if prev_frame is not None:
            # 이전 프레임과 현재 프레임 간 픽셀 값 차이를 grayscale 기준으로 계산
            diff = cv2.absdiff(gray, prev_frame)
            # 픽셀 값 차이를 더하여 보관 (한 개의 스칼라 값이 됨)
            frame_diffs.append((i, diff.sum()))
        prev_frame = gray

    # 가장 큰 변화량을 가진 프레임 선택
    frame_diffs.sort(key=lambda x: x[1], reverse=True)
    selected_frames = sorted([frame_diffs[i][0] for i in range(num_frames)])
    return selected_frames

# 01.18, deamin
def read_frames_cv2(
    video_path: str = None,
    use_segment: bool = True,
    start_time: int = 0,
    end_time: int = 0,
    s3_client: bool = False,
    num_frames: int = 16,
    save_frames_as_img: bool = False,
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
        num_frames: int, 등간격으로 몇 개의 프레임을 추출하여 모델에 넣어줄 지 결정
        save_frames_as_img: bool, 샘플링되는 프레임 저장 여부
    
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
    
    # 단일 segment -> else (01.18, deamin)
    if not use_segment:
        frame_indices: list[int, int] = [start_time * video_fps, end_time * video_fps] # [start_time, end_time]
    else:
        # 전체 frame 사용
        frame_indices: list[int, int] = [0, int(video.get(cv2.CAP_PROP_FRAME_COUNT))]

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
    # print(f"frame_indices: {frame_indices}")
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # print(f"total_frames: {total_frames}, fps: {fps}")

    # segment에 해당하는 frame만 추가
    for idx in sorted(list(np.linspace(frame_indices[0], frame_indices[1]-1, num_frames).astype(int))):#frame_indices[1]): # 등간격
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if not ret:
            raise Exception(f"Failed to read frame: {idx}, Error video path: {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        # 이미지로 저장 (디버깅용 추가: 25.01.25)
        if save_frames_as_img:
            temp_path = os.path.join('./saved_frames', os.path.splitext(os.path.basename(video_path))[0])
            os.makedirs(temp_path, exist_ok=True)
            output_path = os.path.join(temp_path, f"{os.path.basename(video_path)}_{idx:04d}.png")  # 예: frame_0010.png
            cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 저장 시 RGB -> BGR 변환 필요
            print(f"success processing frame idx: {idx} frames")
    video.release()
    

    # (T, H, W, C) to (T, C, H, W), numpy to tensor, dtype=torch.uint8
    frames = torch.tensor(np.stack(frames), dtype=torch.uint8).permute(0, 3, 1, 2)
    return frames, frame_indices, int(total_frames/fps)


def read_frame_cv2(
    video_path: str = None,
    use_segment: bool = True,
    start_time: int = 0,
    end_time: int = 0,
    s3_client: bool = False,
    num_frames: int = 1,
    save_frames_as_img: bool = False,
    # fps: float = 1, # 현재는 동영상 fps를 기준으로 샘플링, 추후 fps를 변경할 때 추가, 
    # sampling: str = 'static', # static, dynamic, dynamic algorithm 토의 후 추가
):
    '''
    단일 segment 대해 frame을 제공하는 함수,
    Video를 받아서 특정 단일 프레임을 tensor로 반환한다.
    
    Args: 
        video_path: str, 동영상 경로
        use_segment: bool, 단일 segment 사용 여부
        start_time: int, 시작 시간
        end_time: int, 종료 시간
        s3_client: bool, s3 클라이언트 사용 여부
        num_frames: int, 각각의 segment에서 몇 개의 프레임을 추출할 지 결정
        save_frames_as_img: bool, 샘플링되는 프레임 저장 여부
    
    Returns:
        frames: torch.Tensor, 이미지 프레임 텐서, (N, C, H, W), torch.uint8
        indices: torch.Tensor, 이미지 프레임 인덱스 텐서, 
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
    frame_index = []
    if not video.isOpened() or video.get(cv2.CAP_PROP_FRAME_COUNT) == 0:
        raise Exception(f"Failed to open video: {video_path}")
    
    # 단일 segment -> else (01.18, deamin)
    if not use_segment:
        frame_indices: list[int, int] = [start_time * video_fps, end_time * video_fps] # [start_time, end_time]
    else:
        # 전체 frame 사용
        frame_indices: list[int, int] = [0, int(video.get(cv2.CAP_PROP_FRAME_COUNT))]


    idx_range = [ int(frame_indices[0] + (frame_indices[1] - frame_indices[0]) * (i / (num_frames + 1)))
        for i in range(1, num_frames + 1) ]
    for idx in sorted(idx_range):
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        # print(f"frame_indices: {frame_indices}")
        
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        # print(f"total_frames: {total_frames}, fps: {fps}")

        # segment에 해당하는 frame만 추가
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if not ret:
            raise Exception(f"Failed to read frame: {idx}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 이미지로 저장 (디버깅용 추가: 25.01.25)
        if save_frames_as_img:
            temp_path = os.path.join('./saved_frames', os.path.splitext(os.path.basename(video_path))[0])
            os.makedirs(temp_path, exist_ok=True)
            output_path = os.path.join(temp_path, f"{os.path.splitext(os.path.basename(video_path))[0]}_{idx:04d}.png")  # 예: frame_0010.png
            cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 저장 시 RGB -> BGR 변환 필요
        frame = torch.tensor(frame, dtype=torch.uint8).permute(2,0,1)
        frames.append(frame)
        frame_index.append(idx)
    video.release()
    
    return frames, frame_index

def read_frame_offset_cv2(
    video_path: str = None,
    use_segment: bool = True,
    start_time: int = 0,
    end_time: int = 0,
    s3_client: bool = False,
    offset: int = 1,
    save_frames_as_img: bool = True,
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
        num_frames: int, 등간격으로 몇 개의 프레임을 추출하여 모델에 넣어줄 지 결정
        offset: int, 프레임을 캡셔닝하기 위해서 주위의 몇 개의 프레임을 추가로 사용할지 결정. [frame_idx-offset, frame_idx+offset]의 범위를 사용
        save_frames_as_img: bool, 샘플링되는 프레임 저장 여부
    
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
    frame_index = []

    if not video.isOpened() or video.get(cv2.CAP_PROP_FRAME_COUNT) == 0:
        raise Exception(f"Failed to open video: {video_path}")
    
    # 단일 segment -> else (01.18, deamin)
    if not use_segment:
        frame_indices: list[int, int] = [start_time * video_fps, end_time * video_fps] # [start_time, end_time]
    else:
        # 전체 frame 사용
        frame_indices: list[int, int] = [0, int(video.get(cv2.CAP_PROP_FRAME_COUNT))]

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
    # print(f"frame_indices: {frame_indices}")
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # print(f"total_frames: {total_frames}, fps: {fps}")

    idx = (frame_indices[0] + frame_indices[-1]) // 2
    # segment에 해당하는 frame만 추가
    for offset_idx in range(idx - offset, idx + offset + 1):
        if 0 <= offset_idx < total_frames:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = video.read()
            if not ret:
                raise Exception(f"Failed to read frame: {idx}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_index.append(offset_idx)

            # 이미지로 저장 (디버깅용 추가: 25.01.25)
            if save_frames_as_img:
                temp_path = os.path.join('./saved_frames', os.path.splitext(os.path.basename(video_path))[0])
                os.makedirs(temp_path, exist_ok=True)
                output_path = os.path.join(temp_path, f"{os.path.basename(video_path)}_{offset_idx:04d}.png")  # 예: frame_0010.png
                cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 저장 시 RGB -> BGR 변환 필요
                print(f"success processing frame idx: {offset_idx} frames")
    video.release()

    # (T, H, W, C) to (T, C, H, W), numpy to tensor, dtype=torch.uint8
    frames = torch.tensor(np.stack(frames), dtype=torch.uint8).permute(0, 3, 1, 2)
    return frames, frame_indices, int(total_frames/fps)

# def HD_transform_padding(frames, image_size=224, hd_num=6):
#     def _padding_224(frames):
#         _, _, H, W = frames.shape

#         # 입력된 높이 H를 224의 배수로 맞추는 작업
#         tar = int(np.ceil(H / 224) * 224)
#         top_padding = (tar - H) // 2
#         bottom_padding = tar - H - top_padding
#         left_padding = 0
#         right_padding = 0

#         padded_frames = F.pad(
#             frames,
#             pad=[left_padding, right_padding, top_padding, bottom_padding],
#             mode='constant', value=255
#         )
#         return padded_frames

#     _, _, H, W = frames.shape
#     trans = False
#     if W < H:
#         frames = frames.flip(-2, -1)
#         trans = True
#         width, height = H, W
#     else:
#         width, height = W, H

#     ratio = width / height
#     scale = 1
#     while scale * np.ceil(scale / ratio) <= hd_num:
#         scale += 1
#     scale -= 1
#     new_w = int(scale * image_size)
#     new_h = int(new_w / ratio)

#     resized_frames = F.interpolate(
#         frames, size=(new_h, new_w),
#         mode='bicubic',
#         align_corners=False
#     )
#     padded_frames = _padding_224(resized_frames)

#     if trans:
#         padded_frames = padded_frames.flip(-2, -1)

#     return padded_frames

# def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
#         best_ratio_diff = float('inf')
#         best_ratio = (1, 1)
#         area = width * height
#         for ratio in target_ratios:
#             target_aspect_ratio = ratio[0] / ratio[1]
#             ratio_diff = abs(aspect_ratio - target_aspect_ratio)
#             if ratio_diff < best_ratio_diff:
#                 best_ratio_diff = ratio_diff
#                 best_ratio = ratio
#             elif ratio_diff == best_ratio_diff:
#                 if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
#                     best_ratio = ratio
#         return best_ratio


# def HD_transform_no_padding(frames, image_size=224, hd_num=6, fix_ratio=(2,1)):
#     min_num = 1
#     max_num = hd_num
#     _, _, orig_height, orig_width = frames.shape
#     aspect_ratio = orig_width / orig_height

#     # calculate the existing video aspect ratio (가능한 Aspect Ratio 전부 추출 및 저장)
#     target_ratios = set(
#         (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
#         i * j <= max_num and i * j >= min_num)
#     # (i,j)가 모인 target_ratios에서 i*j의 값을 순서대로 정렬하여 나타냄
#     target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

#     # find the closest aspect ratio to the target
#     # fix_ratio가 주어졌을 때는 target_aspect_ratio가 fix_ratio가 되고 아니라면 주어진 segments의 H와 W에 따라 가장 가까운 aspect ratio를 구함
#     if fix_ratio:
#         target_aspect_ratio = fix_ratio
#     else:
#         target_aspect_ratio = find_closest_aspect_ratio(
#             aspect_ratio, target_ratios, orig_width, orig_height, image_size)

#     # calculate the target width and height
#     # target_aspect_ratio를 맞춰서 image_size (Default: 224) 에 곱해줌
#     target_width = image_size * target_aspect_ratio[0]
#     target_height = image_size * target_aspect_ratio[1]
#     blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

#     # resize the frames
#     resized_frame = F.interpolate(
#         frames, size=(target_height, target_width),
#         mode='bicubic', align_corners=False
#     )
#     return resized_frame

# def load_video(video_path, num_sampling=2, return_msg=False, resolution=224, hd_num=4, padding=False, save_frames_as_img = True):
#     vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
#     num_frames = len(vr)
#     frame_indices = sorted(np.random.choice(range(0, num_frames), num_sampling, replace=False))
#     # For Saving Image
#     video = cv2.VideoCapture(video_path)
#     for idx in frame_indices:
#         video.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         # print(f"frame_indices: {frame_indices}")
        
#         total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = video.get(cv2.CAP_PROP_FPS)
        
#         # print(f"total_frames: {total_frames}, fps: {fps}")

#         ret, frame = video.read()
#         if not ret:
#             raise Exception(f"Failed to read frame: {idx}")
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # 이미지로 저장 (디버깅용 추가: 25.01.25)
#         if save_frames_as_img:
#             temp_path = os.path.join('./saved_frames', os.path.splitext(os.path.basename(video_path))[0])
#             os.makedirs(temp_path, exist_ok=True)
#             output_path = os.path.join(temp_path, f"{os.path.splitext(os.path.basename(video_path))[0]}_{idx:04d}.png")  # 예: frame_0010.png
#             cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 저장 시 RGB -> BGR 변환 필요

#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)

#     # transform을 통해 224로 Resizing 하지 않음
#     transform = transforms.Compose([
#         transforms.Lambda(lambda x: x.float().div(255.0)),
#         transforms.Normalize(mean, std)
#     ])
#     frames = torch.tensor(vr.get_batch(frame_indices).asnumpy())
#     frames = frames.permute(0, 3, 1, 2)

#     if padding:
#         frames = HD_transform_padding(frames.float(), image_size=resolution, hd_num=hd_num)
#     else:
#         frames = HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)

#     frames = transform(frames)
#     T_, C, H, W = frames.shape

#     sub_img = frames.reshape(
#         1, T_, 3, H//resolution, resolution, W//resolution, resolution
#     ).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, T_, 3, resolution, resolution).contiguous()


#     glb_img = F.interpolate(
#         frames.float(), size=(resolution, resolution), mode='bicubic', align_corners=False
#     ).to(sub_img.dtype).unsqueeze(0)

#     frames = torch.cat([sub_img, glb_img])


#     if return_msg:
#         fps = float(vr.get_avg_fps())
#         sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
#         # " " should be added in the start and end
#         msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
#         return frames, msg
#     else:
#         return frames, frame_indices