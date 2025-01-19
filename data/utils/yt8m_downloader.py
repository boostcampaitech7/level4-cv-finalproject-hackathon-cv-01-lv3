from typing import List, Optional
import os
import subprocess
import tensorflow as tf
from threading import Thread
import time
from pathlib import Path

# MovieClipID : 734
wanted_video_id = 734

def process_shard(shard_num: int, 
                 total_shards: int = 3844, 
                 wanted_video_id: int = 734, 
                 partition: str = '2/frame/train', 
                 save_dir: Optional[str] = None) -> None:
    """YT8M 데이터셋의 특정 샤드를 다운로드하고 원하는 video_id가 있는 파일만 저장

    Args:
        shard_num: 처리할 샤드 번호
        total_shards: 전체 샤드 수
        wanted_video_id: 찾고자 하는 비디오 ID
        partition: 데이터 파티션 (예: '2/frame/train')
        save_dir: 저장할 디렉토리 경로
    """
    # 환경 변수 설정(두 번째 예제와 동일)
    env = os.environ.copy()
    env['shard'] = f'{shard_num},{total_shards}'
    env['partition'] = partition
    env['mirror'] = 'asia'

    try:
        # data.yt8m.org/download.py를 받아와 파이썬으로 실행
        curl_proc = subprocess.Popen(['curl', 'data.yt8m.org/download.py'], 
                                     stdout=subprocess.PIPE)
        python_proc = subprocess.Popen(['python'], 
                                       stdin=curl_proc.stdout,
                                       env=env)
        
        curl_proc.stdout.close()
        python_proc.communicate()

        print(f"Successfully processed shard {shard_num}")
    except Exception as e:
        print(f"Failed to process shard {shard_num}: {str(e)}")

def process_file(filename: str, wanted_video_id: int) -> List[str]:
    """TFRecord 파일에서 원하는 video_id를 검색하고 찾은 ID 목록 반환

    Args:
        filename: TFRecord 파일 경로
        wanted_video_id: 찾고자 하는 비디오 ID

    Returns:
        찾은 video_id 목록
    """
    found_video_ids = []
    try:
        for record in tf.data.TFRecordDataset(filename):
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            
            # 실제 비디오 ID 키는 "id" 또는 "video_id" 등으로 데이터셋마다 다를 수 있음
            # 라벨은 "labels" 키에 들어있는 경우가 많음
            video_id = example.features.feature["id"].bytes_list.value[0].decode("utf-8")
            label_list = example.features.feature["labels"].int64_list.value

            if wanted_video_id in label_list:
                found_video_ids.append(video_id)
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")
        return []
    
    return found_video_ids

def parallel_process(num_threads: int = 4, 
                    total_shards: int = 3844, 
                    partition: str = '2/frame/train', 
                    save_dir: Optional[str] = None) -> None:
    """여러 스레드를 사용하여 YT8M 데이터셋을 병렬로 다운로드하고 처리

    Args:
        num_threads: 사용할 스레드 수
        total_shards: 전체 샤드 수
        partition: 데이터 파티션
        save_dir: 저장할 디렉토리 경로
    """
    threads = []
    
    start_time = time.time()  # 시작 시간 기록
    
    for i in range(1, total_shards + 1):
        thread = Thread(
            target=process_shard,
            args=(i, total_shards, wanted_video_id, partition, save_dir)
        )
        threads.append(thread)
        thread.start()
       
        # 지정된 개수만큼 스레드가 생성되면 모두 완료될 때까지 대기
        if len(threads) >= num_threads:
            for t in threads:
                t.join()
            threads = []  # 스레드 리스트 초기화
            
            # 현재 디렉토리에 존재하는 .tfrecord 파일 확인
            filenames = [
                f for f in os.listdir('.') if f.endswith('.tfrecord')
            ]
            
            for filename in filenames:
                found_video_ids = process_file(filename, wanted_video_id)
                # wanted_video_id가 하나도 없으면 삭제
                if not found_video_ids:
                    os.remove(filename)

    # 마지막 그룹의 스레드들도 모두 완료될 때까지 대기
    for t in threads:
        t.join()

    # 혹시 마지막에 남은 파일도 검증
    filenames = [
        f for f in os.listdir('.') if f.endswith('.tfrecord')
    ]
    for filename in filenames:
        found_video_ids = process_file(filename, wanted_video_id)
        if not found_video_ids:
            os.remove(filename)

    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time
    print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

if __name__ == '__main__':
    NUM_THREADS = 8     # 동시 다운로드 및 처리 수
    TOTAL_SHARDS = 3844   # 전체 샤드 수
    # frame or segment
    is_frame = True
    # 데이터 파티션 (2/frame/train, 2/frame/validate, 2/frame/test,3/frame/validate, 3/frame/test)
    PARTITION = '2/frame/train' if is_frame else '3/frame/validate'
    
    # 저장 경로 설정 (상대 경로 사용)
    SAVE_DIR = "./data/yt8m/frame" if is_frame else "./data/yt8m/segment"
    
    # 저장 디렉토리 생성 및 이동
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.chdir(SAVE_DIR)
    
    parallel_process(NUM_THREADS, TOTAL_SHARDS, PARTITION, SAVE_DIR)
