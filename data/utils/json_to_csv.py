import os
import pandas as pd
import json

def label_to_video(label: str) -> str:
        '''
        label의 경로에 따라 그에 맞는 video의 경로를 반환
        '''
        replacements = {'json':'mp4', 'labels':'clips'}
        video = label # To be transformed
        for old, new in replacements.items():
            video = video.replace(old, new)
        return video

def integrity_check(labels, clips):
        '''
        모든 labels가 clips와 매칭되는 지 확인합니다
        '''
        transformed_labels = [label_to_video(x) for x in labels]
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
        
def load_label(data_path: str, train: str) -> list:
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
        integrity_check(all_labels, all_clips)
        return all_labels

def make_csv(labels: str, dest: str='.'):
    '''
    labels를 받아 통합 csv형태로 반환
    segment_name=row['segment_name'],
                start_time=row['start_time'],
                end_time=row['end_time'],
                caption=row['caption'],
                caption_ko=row['caption_ko'],
    '''
    meta = pd.DataFrame(columns=['segment_name', 'start_time', 'end_time', 'caption', 'caption_ko'])
    for label in labels:
        with open(label, encoding='utf-8') as f:
            data = json.load(f)
            segment_name = os.path.splitext(os.path.basename(label))[0]
            start_time = data[segment_name]['start_time']
            end_time = data[segment_name]['end_time']
            caption = data[segment_name]['caption']
            caption_ko = data[segment_name].get('caption_ko', "")
            new_row = pd.DataFrame([{'segment_name': segment_name, 'start_time': start_time, 'end_time': end_time, 'caption': caption, 'caption_ko': caption_ko}])
        meta = pd.concat([meta, new_row], ignore_index=True)
    meta.to_csv(os.path.join(dest, f"metadata.csv"), index=False, encoding='utf-8')

labels = load_label(data_path = '../../../../data/data', train='train')
make_csv(labels, '.')
# print(labels)