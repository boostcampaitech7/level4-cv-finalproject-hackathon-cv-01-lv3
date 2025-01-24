#pip install --upgrade scenedetect
#pip install ffmpeg-python
import json
import os
import math
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector

def split_video_into_scenes(video_path : str, threshold:float = 27.0, output_json_dir : str ="./video", segments_dir : str ="./segments"):
    """
    비디오를 장면(scene,segment)별로 분할하고 각 장면에 대한 메타데이터(seg_id, start_time, end_time, caption)를 JSON 파일로 저장합니다.

    Args:
        video_path (str): 처리할 비디오 파일의 경로
        threshold (float, optional): 장면 감지를 위한 임계값. 기본값은 27.0
        output_json_dir (str, optional): JSON 메타데이터 파일을 저장할 디렉토리 경로. 기본값은 "./video"
        segments_dir (str, optional): 분할된 비디오 세그먼트를 저장할 디렉토리 경로. 기본값은 "./segments"

    Returns:
        None

    Raises:
        FileNotFoundError: 비디오 파일이 존재하지 않는 경우
        PermissionError: 디렉토리 생성 권한이 없는 경우
    """
    #scene detection
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    #video_id 찾기
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(segments_dir, exist_ok=True)    # segment 저장 디렉터리 생성

    # 비디오 segment  저장
    split_video_ffmpeg(
        video_path,
        scene_list,
        output_dir=segments_dir,
        output_file_template=f"{video_id}_$SCENE_NUMBER.mp4",
        arg_override='-map 0', #ffmpeng 에러 방지
        show_progress=True,
        # show_output=True
    )
    print(f"비디오 세그먼트 저장 완료: {segments_dir}")
    total_scenes = len(scene_list)
    if not total_scenes:
        print(f"{video_id} 에서 발생한 세그먼트가 없습니다.")
        return
    digit_count = max(3, math.floor(math.log10(total_scenes)) + 1)

    for idx, scene in enumerate(scene_list):

        formatted_idx = f"{idx+1:0{digit_count}d}"  # 0으로 패딩된 동적 자리수 , idx는 1부터 시작
        seg_id = f"{video_id}_{formatted_idx}" # seg_id = video_id + idx(:3d)
        start_time = scene[0].get_timecode().split(".")[0] # hh:mm:ss
        end_time = scene[1].get_timecode().split(".")[0] # hh:mm:ss

        # 개별 segment의 JSON 내용 초기화
        json_data = {
            seg_id : {
              "start_time": start_time,
              "end_time": end_time,
              "caption": "nan"
            }
        }

        # JSON 데이터 저장
        os.makedirs(output_json_dir, exist_ok=True)
        seg_json_path = os.path.join(output_json_dir, f"{seg_id}.json")
        with open(seg_json_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        # print(f"{idx}'s seg JSON 파일 저장 완료: {seg_json_path}")

# def main():
#     video_path ="./video/video1.mp4"  # video_path
#     output_json_dir = "./segments_anno"  # seg json 저장 폴더
#     segments_dir = "./segments"  # seg.mp4 저장 폴더
#     split_video_into_scenes(video_path, threshold=27.0, output_json_dir=output_json_dir, segments_dir=segments_dir)

# if __name__ == "__main__":
#     main()
