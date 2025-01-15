import tensorflow as tf
import torch
import pandas as pd
import glob
import os
import numpy as np

from torch.utils.data import DataLoader

class YoutubeSegmentDataset(torch.utils.data.IterableDataset):
    """
    Youtube-8M-Segment Dataset을 처리할 수 있는 데이터셋 클래스입니다.
    
    args
    file_paths: .tfrecord 형식의 Youtube-8M-Segment Dataset의 데이터 경로를 설정하세요
    seed(Optional): 시드 설정을 위해 입력받으며, 입력하지 않을 시 Default 42로 설정됩니다
    debug(Optional): 디버깅을 수행하려면 True로 설정하세요. Default: False
    vocab_path (Optional): segment_vocabulary.csv의 경로를 입력하세요. Default: ./data/segment_vocabulary.csv
    epochs (Optional): 병렬화를 할 때, 예외처리를 위함 (epochs가 1인데, num_workers가 1이상이 되는 경우 방지). 만약, epochs가 1이라면 Validation으로 취급합니다
    max_examples (Optional): 데이터 셋 내에서 데이터의 수량을 제한하고자 할 때 사용합니다. None으로 설정 시, 제한하지 않고 전체를 이용하겠다는 의미입니다. Default: None
    offset (Optional): Video feature 앞 뒤로 얼마나 padding을 적용할 것인지를 설정합니다. Default: 0
    """
    def __init__(self, file_paths, seed=42, debug=False,
                 vocab_path="./data/segment_vocabulary.csv",
                 epochs=1, max_examples=None, offset=0):
        super(YoutubeSegmentDataset).__init__()
        self.file_paths = list(glob.glob(str(file_paths))) if "*" in file_paths else file_paths
        self.seed = seed
        self.debug = debug
        self.max_examples = max_examples
        vocab = pd.read_csv(vocab_path)
        self.label_mapping = {label: index for label, index in zip(vocab["Index"], vocab.index)}
        self.name_mapping = {index: name for index, name in zip(vocab.index, vocab['Name'])}
        # segment_vocabulary는 어떻게 모은거지?
        self.epochs = epochs
        # offset의 용도는?
        self.offset = offset

    def __iter__(self):
        """
        데이터 생성기를 반환
        """
        worker_info = torch.utils.data.get_worker_info() # 단일 프로세스의 경우, None이 출력됨
        if worker_info is None:
            seed = self.seed
        else:
            if worker_info.num_workers > 1 and self.epochs == 1: # 병렬화 시, epochs = 1인 경우 num_workers가 2 이상 되는 경우를 방지하기 위함 (데이터 분배와 반복 처리가 충돌 가능성이 있음)
                raise ValueError("Validation cannot have num_workers > 1")
            seed = self.seed + worker_info.id # 이건 왜 이렇게 하는지 이해 안됨
        return self.generator(seed)

    def prepare_one_sample(self, row):
        """
        특정 row, 즉 데이터에 접근하면 그에 맞는 feature (frame, segment_label)을 반환해주는 함수
        """
        example = tf.train.SequenceExample()
        tmp = example.FromString(row.numpy())
        context, video_features = tmp.context, tmp.feature_lists

        vid_labels = list(context.feature['labels'].int64_list.value)
        vid_labels_encoded = set([self.label_mapping[x] for x in vid_labels if x in self.label_mapping])

        # 뽑을 feature들 따로 저장
        segment_labels = np.array(context.feature['segment_labels'].int64_list.value).astype("int64")
        segment_start_times = np.array(context.feature['segment_start_times'].int64_list.value)
        segment_scores = np.array(context.feature['segment_scores'].float_list.value).astype("int64")
        vid = context.feature['id'].bytes_list.value[0].decode('utf8')


        # Transform label
        segment_labels = np.array([self.label_mapping[x] for x in segment_labels])

        # Dummy Data. Segment Label을 인덱스->영문으로 변환하여 저장. 사용 중이지는 않음
        segment_labels_name = np.array(list(set([self.name_mapping[x] for x in segment_labels])))

        # Negative Mining (우리 Task에서 굳이 필요할까?)
        if self.debug:
            if not vid_labels_encoded: # Set이 공집합이다 == 인코딩이 잘 되지 않았을 것
                print(segment_labels, vid_labels)
            else:
                print("Passed")
        negative_mask = np.zeros(1000, dtype=int)
        # segment_labels에 없는 레이블을 1로 지정함
        negative_mask[np.array(list(set(range(1000)) - vid_labels_encoded - set(segment_labels)))] = 1

        # Frames. Shape: (frames, 1024)
        tmp = video_features.feature_list['rgb'].feature
        frames = tf.cast(tf.io.decode_raw([x.bytes_list.value[0] for x in tmp], out_type='uint8'), "float32").numpy()

        # Audio. Shape: (frames, 128) - 추후 사용 예정
        tmp = video_features.feature_list['audio'].feature
        audio = tf.cast(tf.io.decode_raw([x.bytes_list.value[0] for x in tmp], out_type="uint8"), "float32").numpy()

        # Frames + Audio Combined: (frames, 1024+128=1152) - 추후 사용 예정
        video_features = torch.from_numpy(frames)
        # video_features = torch.from_numpy(np.concatenate([frames, audio], axis=-1))

        if self.debug:
            print(f"http://data.yt8m.org/2/j/i/{vid[:2]}/{vid}.js")
            print(segment_labels)
            print(segment_start_times)
            print(segment_scores)
            print(video_features.size(0))
            print("=" * 20 + '\n')

        # skip problematic entries
        assert segment_start_times.max() <= video_features.size(0)

        video_features_padded = torch.cat(
            [
                torch.zeros(self.offset, video_features.size(1), dtype=video_features.dtype),
                video_features,
                torch.zeros(5 + self.offset, video_features.size(1), dtype=video_features.dtype)
            ],
            dim=0
        )

        # Create segments. Shape: (n_segments, 5 + self.offset * 2) - 세그먼트의 시간 인덱스
        indices = (
            torch.from_numpy(segment_start_times).unsqueeze(1) +
            torch.arange(5 + 2 * self.offset).unsqueeze(0)
        )

        # Shape: (n_segments * (5 + self.offset * 2), 1152) -> (n_segments, 5 + self.offset * 2, 1152)
        # 이 코드 정확히는 이해 안됨
        segments = torch.index_select(video_features_padded, 0, indices.view(-1)).view(indices.size(0), 5 + self.offset * 2, -1)

        return video_features, segments, torch.from_numpy(segment_labels).unsqueeze(1), segment_labels_name, torch.from_numpy(segment_scores).unsqueeze(1)

    def _iterate_through_dataset(self, tf_dataset):
        """
        tf_dataset을 받아 데이터를 반환해주는 내부 함수
        """
        ## 임시
        for row in tf_dataset:
            # self.prepare_one_sample(row)를 한 번 더 감싸서 튜플 형태로 만들어 Rank를 1 이상으로 만들어 주기 위함 (안하면 Value Error - Unbatching a tensor 오류뜸)
            video_features, segments, segment_labels, segment_labels_names, segment_scores = self.prepare_one_sample(row)
            for segment, label, n_label, score in zip(segments, segment_labels, segment_labels_names, segment_scores):
                yield video_features, segment, torch.cat([label, score]), n_label


    def generator(self, seed):
        """
        데이터셋을 의미함. Seed 설정에 맞추어 사용자의 편의에 따라 데이터셋을 운용할 수 있음
        """
        if self.epochs == 1:
            # Validation
            tf_dataset = tf.data.TFRecordDataset(tf.data.Dataset.from_tensor_slices(self.file_paths))
        
        else:
            # tf_dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files("./*.tfrecord"))
            tf_dataset = tf.data.TFRecordDataset(tf.data.Dataset.from_tensor_slices(self.file_paths).shuffle(100, seed=seed, reshuffle_each_iteration=True)).repeat(self.epochs).shuffle(256, seed=seed, reshuffle_each_iteration=True).repeat(self.epochs)
        
        for n_example, row in enumerate(self._iterate_through_dataset(tf_dataset)):
            if self.max_examples and self.max_examples == n_example:
                break
            yield row

def collate_videos(batch, pad=0):
    """
    Batch Preparation
    Pads the sequences if needed
    """
    transposed = list(zip(*batch))
    max_len = max(len(x) for x in transposed[0])
    data = torch.zeros((len(batch), max_len, transposed[0][0].size(-1)), dtype=torch.float) + pad
    masks = torch.zeros((len(batch), max_len), dtype=torch.float) # padding부분을 제외시켜주기 위함
    for i, row in enumerate(transposed[0]):
        data[i, :len(row)] = row
        masks[i, :len(row)] = 1
    
    # Labels
    if transposed[1][0] is None:
        return data, masks, None
    labels = torch.stack(transposed[1]).float()
    return data, masks, labels

file_path = "./*.tfrecord"
dataset = YoutubeSegmentDataset(file_paths=file_path)
for i,(v,s,ls,nl) in enumerate(dataset):
    print(f"--------------{i} Data --------------")
    print(f"video_features.shape: {v.shape}")
    print(f"segment.shape: {s.shape}")
    print(f"label+score.shape: {ls.shape}, segment_label: {ls}")
    print(f"named label: {nl}")
    print("--------------------------------------")
    if i==3:
        break

loader = DataLoader(dataset, num_workers=0, batch_size=16, collate_fn = collate_videos)

for i, (data, masks, labels) in enumerate(loader):
    print(f"NOW: {(i+1)*2}")
    print(data.shape)
    print(masks.shape)
    print(labels.shape)