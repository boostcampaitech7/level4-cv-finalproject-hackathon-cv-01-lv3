import pandas as pd
import os

def cut(segment_name: str, caption: str, start_time: str, end_time: str, caption_ko: str):
    captions = caption.split('.')
    new_rows = []
    for i, cap in enumerate(captions):
        if len(cap.strip()):
            # if cap.strip()[0] != '"':
            #     print(f"cap does not starts with apostrophe {cap}, {cap.strip()[0]}")
            #     cap = '"' + cap.strip()
            # if cap.strip()[-1] != '"':
            #     print(f"cap does not ends with apostrophe {cap}, {cap.strip()[-1]}")
            #     cap = cap.strip() + '"'
            new_rows.append([segment_name, i, start_time, end_time, cap.strip(), caption_ko])
    return new_rows

def save_csv(csv_path: str, dest: str):
    meta = pd.read_csv(csv_path, encoding='utf-8')
    new_meta = []
    for i in range(len(meta)):
        row = meta.iloc[i, :]
        segment_name = row['segment_name']
        caption = row['caption']
        start_time = row['start_time']
        end_time = row['end_time']
        caption_ko = row['caption_ko']
        
        new_ = cut(segment_name, caption, start_time, end_time, caption_ko)
        new_meta.extend(new_)  # 리스트로 추가

    # DataFrame으로 변환 후 저장
    new_meta_df = pd.DataFrame(new_meta, columns=['segment_name', 'sentence_index', 'start_time', 'end_time', 'caption', 'caption_ko'])
    new_meta_df.to_csv(dest, index=False, encoding='utf-8')
        
save_csv('./data/utils/metadata.csv', './data/utils/one_sentence_metadata.csv')