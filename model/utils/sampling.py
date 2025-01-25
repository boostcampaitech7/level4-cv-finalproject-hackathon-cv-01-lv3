import cv2
import numpy as np
import torch

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.asarray(gray, dtype=np.uint8)
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
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
    num_frames: int = 8,
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
    else:
        print(f"Successfully opened video: {video_path}")
    
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
    for idx in np.linspace(frame_indices[0], frame_indices[1]-1, num_frames).astype(int):#frame_indices[1]): # 등간격
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if not ret:
            raise Exception(f"Failed to read frame: {idx}")
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