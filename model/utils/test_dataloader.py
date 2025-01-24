import os
import torch
from .data_utils import InternVideo2_VideoChat2_Dataset, InternVideo2_VideoChat2_DataLoader

def test_dataset_and_loader(
    csv_path: str = None,
    video_root: str = None,
    batch_size: int = 2,
    use_audio: bool = False,
    num_workers: int = 0  # 디버깅을 위해 0으로 설정
):
    print("===== 데이터셋 테스트 시작 =====")
    
    # 데이터셋 초기화
    try:
        dataset = InternVideo2_VideoChat2_Dataset(
            csv_path=csv_path,
            video_root=video_root,
            use_segment=True,
            use_audio=use_audio,
            train=True
        )
        print(f"데이터셋 생성 성공: 총 {len(dataset)}개 샘플")
    except Exception as e:
        print(f"데이터셋 생성 실패: {str(e)}")
        return

    # 데이터로더 초기화
    try:
        dataloader = InternVideo2_VideoChat2_DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            use_audio=use_audio
        )
        print(f"데이터로더 생성 성공: 총 {len(dataloader)}개 배치")
    except Exception as e:
        print(f"데이터로더 생성 실패: {str(e)}")
        return

    # 첫 번째 배치 테스트
    print("\n첫 번째 배치 테스트 중...")
    try:
        batch = next(iter(dataloader))
        
        # 배치 정보 출력
        print(f"\n배치 구조:")
        print(f"- frames shape: {batch['frames'].shape}")
        if use_audio:
            print(f"- audio shape: {batch['audio'].shape}")
        print(f"- segment names: {batch['segment_names']}")
        print(f"- annotations: {batch['annotations']}")
        
        # 텐서 값 범위 확인
        frames = batch['frames']
        print(f"\n프레임 텐서 정보:")
        print(f"- dtype: {frames.dtype}")
        print(f"- value range: [{frames.min():.3f}, {frames.max():.3f}]")
        print(f"- mean: {frames.mean():.3f}")
        print(f"- std: {frames.std():.3f}")
        
    except Exception as e:
        print(f"배치 로딩 실패: {str(e)}")
        return

    print("\n===== 데이터셋 테스트 완료 =====")

if __name__ == "__main__":
    # 테스트 설정
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # 상위 디렉토리로 이동하여 필요한 경로 생성
    video_path = os.path.join(current_dir, "data/clips")
    csv_path = os.path.join(current_dir, "data/labels")

    test_dataset_and_loader(
        csv_path=csv_path,
        video_root=video_path,
        batch_size=2,
        use_audio=False,
        num_workers=0
    ) 