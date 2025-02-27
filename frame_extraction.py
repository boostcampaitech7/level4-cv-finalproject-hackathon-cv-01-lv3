from model.utils.data_utils_from_json import InternVideo2_VideoChat2_Dataset

dataset = InternVideo2_VideoChat2_Dataset('../../../../../dev/shm/Cleaned_final_split', save_frames_as_img=True, save_dir='../../../../../dev/shm/saved_frames', resize=448, num_frames=4)
for x in dataset:
    pass




# For Testing / Debugging
# import cv2
# import numpy as np
# import os

# video_path = "../../../../../dev/shm/Cleaned_final_split/clips/yt8m_Movieclips_QoYlqdBtRSw_089.mp4"
# cap = cv2.VideoCapture(video_path)

# save_dir = './'

# frames = []
# if not cap.isOpened():
#     print(f"⚠️ 비디오를 열 수 없습니다: {video_path}")
# else:
#     ret, frame = cap.read()
#     if not ret:
#         print(f"⚠️ 첫 번째 프레임을 읽을 수 없습니다: {video_path}")
#     else:
#         print("✅ 비디오가 정상적으로 열립니다.")
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_indices: list[int, int] = [0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
#     print(f"total_frames: {total_frames}")
#     print(f"fps: {fps}")
#     for idx in sorted(list(np.linspace(frame_indices[0], frame_indices[1]-1, 4).astype(int))):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
#         if not ret:
#             raise Exception(f"Failed to read frame: {idx}, Error video path: {video_path}")
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(frame)
#         # 이미지로 저장 (디버깅용 추가: 25.01.25)
#         temp_path = os.path.join(save_dir, os.path.splitext(os.path.basename(video_path))[0])
#         os.makedirs(temp_path, exist_ok=True)
#         output_path = os.path.join(temp_path, f"{os.path.basename(video_path)}_{idx:04d}.png")  # 예: frame_0010.png
#         cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 저장 시 RGB -> BGR 변환 필요
#         print(f"success processing frame idx: {idx} frames")
# cap.release()
