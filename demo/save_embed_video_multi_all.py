import torch
from LongCLIP.model import longclip
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import os
import torch
from elasticsearch import Elasticsearch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm
from torch import Tensor
from torch.cuda.amp import autocast
import numpy as np

INDEX_NAME = "newnew_clip"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_elasticsearch_client():
    elasticsearch_url = os.environ.get('ELASTICSEARCH_URL')
    elasticsearch_api_key = os.environ.get('ELASTICSEARCH_API_KEY')
    
    if not elasticsearch_url or not elasticsearch_api_key:
        raise ValueError("환경 변수 ELASTICSEARCH_URL과 ELASTICSEARCH_API_KEY를 설정해주세요.")
    return Elasticsearch(
        elasticsearch_url,
        api_key=elasticsearch_api_key
    )

class VideoCaption:
    def __init__(self):
        # 설정 로드
        self.model, self.preprocess = longclip.load("./LongCLIP/checkpoints/longclip-L.pt", DEVICE)
    
    def sample_video_frames(self, video_path, num_frames=8, target_size=(224,224)): 
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 균등 간격으로 8개 프레임 선택 
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                frame = cv2.resize(frame, target_size)  # 프레임 크기 조정
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB 변환
                frames.append(frame)

        cap.release()
        print(f"total frames: {total_frames}")
        
        return np.array(frames)  # (16, 224, 224, 3) 형태 반환

    def encode_video(self, video_path):
        frames = self.sample_video_frames(video_path)  # (16, 224, 224, 3)

        # 프레임을 PIL 이미지로 변환한 후 preprocess 적용
        frames_pil = [Image.fromarray(frame) for frame in frames]  # numpy.ndarray -> PIL.Image
        frames_tensor = torch.stack([self.preprocess(frame) for frame in frames_pil]).to(DEVICE)  # (16, 3, 224, 224)
        
        # 각 프레임을 개별적으로 인코딩하고 평균내기
        with torch.no_grad():
            frame_features = [self.model.encode_image(frame.unsqueeze(0)) for frame in frames_tensor]
        
        # 각 프레임의 특징 벡터를 평균하여 비디오 임베딩 생성
        video_embedding = torch.mean(torch.stack(frame_features), dim=0)
        
        # 각 프레임의 임베딩을 정규화
        frame_embeddings = [{"frame_index": i, 
                            "frame_embedding": (frame_feature.squeeze() / frame_feature.norm(dim=-1, keepdim=True))}#.tolist()} 
                            for i, frame_feature in enumerate(frame_features)]
        
        # 비디오 임베딩도 정규화
        video_embedding = video_embedding / video_embedding.norm(dim=-1, keepdim=True)
        
        return video_embedding, frame_embeddings

    def encode_text(self, text):
        text_input = longclip.tokenize([text]).to(DEVICE)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def generate_embedding(self, video_path):
        """embedding 생성"""
        video_embedding, frame_embeddings = self.encode_video(video_path)
        return video_embedding, frame_embeddings
    
    def save_to_elasticsearch(self, client, segment_name, start_time, end_time, caption, caption_ko, video_embedding, frame_embeddings, text_embedding = None, index_name=INDEX_NAME):
        """임베딩을 Elasticsearch에 저장
            Text Embedding은 확인용도로, 실제 계산 시에는 쓰이지 않을 예정임
        """

        doc = {
            'segment_name': segment_name,
            'start_time': start_time,   
            'end_time': end_time,
            'caption': caption,
            'caption_ko': caption_ko,
            'video_embedding': video_embedding.squeeze().tolist(),
            'frame_embeddings': frame_embeddings,  # 각 프레임의 임베딩 추가
            'caption_embedding' : text_embedding.squeeze().tolist() if text_embedding is not None else ""
        }
        
        try:
            response = client.index(index=index_name, document=doc)
            print(f"Document indexed: {response['result']}")
            print(f"Embedding dimension: {len(doc['video_embedding'])}")
        except Exception as e:
            print(f"Error saving to Elasticsearch: {str(e)}")
            print(f"Video Embedding shape: {video_embedding.shape}")

def create_index():
    client = get_elasticsearch_client()
    index_name = INDEX_NAME
    
    # 기존 인덱스가 있다면 삭제
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
        print(f"Deleted existing index: {index_name}")
    
    # 새로운 인덱스 생성 with mappings
    mappings = {
        "mappings": {
            "properties": {
                "segment_name": {"type": "keyword"},
                "start_time": {"type": "keyword"},
                "end_time": {"type": "keyword"},
                "caption": {"type": "text"},
                "caption_ko": {"type": "text"},
                "video_embedding": {"type": "dense_vector", "dims": 768},
                "frame_embeddings": {
                    "type": "nested",  # Nested 타입으로 설정
                    "properties": {
                        "frame_index": {"type": "integer"},
                        "frame_embedding": {"type": "dense_vector", "dims": 768}
                    }
                },
                "caption_embedding": {"type": "dense_vector", "dims": 768}
            }
        }
    }
    
    client.indices.create(index=index_name, body=mappings)
    print(f"Created new index: {index_name} with mappings")

def save_from_csv(csv_path: str, data_path: str = '../../video'):
    client = get_elasticsearch_client()
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    captioner = VideoCaption()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing", unit="row"):
        if pd.isna(row['caption']):
            print(f"Skipping {row['segment_name']}: Caption is NaN")
            continue
        # 이미 존재하는 segment_name인지 확인
        search_result = client.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "term": {
                        "segment_name": row['segment_name']
                    }
                }
            }
        )
        
        # 이미 존재하는 경우 건너뛰기
        if search_result['hits']['total']['value'] > 0:
            print(f"Skipping {row['segment_name']}: Already exists")
            continue
        
        video_path = get_video(video_name=row['segment_name'], data_path=data_path)
        if video_path is None:
            print(f"passing {row['segment_name']}")
            continue
        
        try:
            # 임베딩 생성
            video_embedding, frame_embeddings = captioner.generate_embedding(video_path)
            text_embedding = captioner.encode_text(row['caption'])
            frame_embeddings_list = [{"frame_index": emb['frame_index'], "frame_embedding": emb['frame_embedding'].squeeze().tolist()} for emb in frame_embeddings]

            # Elasticsearch에 저장
            captioner.save_to_elasticsearch(
                client=client,
                segment_name=row['segment_name'],
                start_time=row['start_time'],
                end_time=row['end_time'],
                caption=row['caption'],
                caption_ko="",
                video_embedding=video_embedding,
                frame_embeddings=frame_embeddings_list,
                text_embedding=text_embedding,
                index_name=INDEX_NAME
            )
            print(f"Successfully processed: {row['segment_name']}")
            
        except Exception as e:
            print(f"Error processing {row['segment_name']}: {str(e)}")
            continue

def search_videos(query_text: str):
    start = time.time()
    print(f"start_time : {start}")
    client = get_elasticsearch_client()
    print(f"server started")
    
    # 쿼리 텍스트를 임베딩 벡터로 변환
    model = VideoCaption()
    query_embedding = model.encode_text(query_text)
    
    # 검색 쿼리 구성: frame_embeddings 내에서 각각의 프레임 임베딩과 비교
    search_query = {
        "size": 1,  # 상위 1개 결과 반환
        "query": {
            "nested": {
                "path": "frame_embeddings",  # frame_embeddings 배열을 기준으로 검색
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},  # 모든 프레임에 대해 유사도 계산
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'frame_embeddings.frame_embedding') + 1.0",
                            "params": {"query_vector": query_embedding.squeeze().tolist()}
                        }
                    }
                },
                "inner_hits": {
                    "size": 1  # 유사도가 가장 높은 프레임을 반환
                }
            }
        }
    }
    
    # 검색 실행
    retrieve = []
    try:
        start_search = time.time()
        print(f"start search: {start_search}")
        results = client.search(index=INDEX_NAME, body=search_query)
        print("\n1 Stage Search Results:")
        
        max_score = -1  # 유사도 값 중 가장 높은 값을 찾기 위한 변수
        best_segment_name = None
        
        # 각 히트에 대해 가장 유사도가 높은 프레임을 찾아서, 해당 프레임을 포함한 비디오의 segment_name을 반환
        for hit in results['hits']['hits']:
            source = hit['_source']
            inner_hits = hit['inner_hits']['frame_embeddings']['hits']['hits']
            
            # 프레임 내에서 가장 높은 유사도를 가진 프레임을 선택
            if inner_hits:
                best_frame = inner_hits[0]
                score = best_frame['_score']  # 가장 높은 유사도 값
                if score > max_score:
                    max_score = score
                    best_segment_name = source['segment_name']  # 가장 높은 유사도 값을 가진 segment_name을 저장
        
        if best_segment_name:
            print(f"\nBest Segment Name: {best_segment_name}")
            print(f"Highest Similarity Score: {max_score}")
            retrieve.append({
                'Segment_name': best_segment_name,
                'Similarity Score': max_score
            })

        end = time.time()
        embedding_time = start_search - start
        searching_time = end - start_search
        total_time = end - start
        print(f"Embedding: {start_search - start: .3f}seconds!")
        print(f"Searching: {end - start_search: .3f}seconds!")
        print(f"Embedding + Searching: {end - start: .3f}seconds!")
        
    except Exception as e:
        print(f"Error during search: {str(e)}")
    
    return retrieve, embedding_time, searching_time, total_time

def run(query_text: str)-> tuple[str, str]:
    results, segment_name = search_videos(query_text)

    print(f"\n\n[DEBUG] results: {results}")
    print(f"[DEBUG] segment_name: {segment_name}")
    return results, segment_name

def find_video(data_path: str, segment_name: str):  
    """
    Demo 용
    """
    dsrc, category, _, _ = segment_name.split('_')
    video_path = os.path.join(data_path, dsrc.upper(), category)
    for x in ['train', 'test']:
        if os.path.exists(os.path.join(video_path, x, 'clips', f'{segment_name}.mp4')):
            return os.path.join(video_path, x, 'clips', f'{segment_name}.mp4')

def get_video(video_name: str, data_path: str = '../../video'):
    """
    T2V 시, CLIP에서 사용하는 용도
    """
    video_path = os.path.join(data_path, 'clips', f"{video_name}.mp4")
    if os.path.exists(video_path):
        return video_path
    else:
        print(f"We could not find the video {video_name} in {video_path}! Returning None instead")
        return None
    
# R@1 계산 함수
def calculate_r1_r5_r10(csv_path):
    df = pd.read_csv(csv_path)
    evaluations = pd.DataFrame(columns=['Query_Type', 'Query', 'Prediction', 'GT', 'R@1', 'R@5', 'R@10', 'embedding_time', 'searching_time', 'total_time'])
    
    correct_count_r1 = 0
    correct_count_r5 = 0
    correct_count_r10 = 0
    correct_video_count_r1 = 0
    correct_video_count_r5 = 0
    correct_video_count_r10 = 0
    total_queries = len(df)

    embedding_times = []
    searching_times = []
    total_times = []

    for _, row in tqdm(df.iterrows(), total=total_queries, desc="Calculating R@1, R@5, R@10", unit="query"):
        query = row["Generated_Query"]
        query_type = row['Query_Type']
        true_segment_name = row["segment_name"]  # 실제 정답 segment_name
        
        predicted_segments, embedding_time, searching_time, total_time = search_videos(query)  
        r1, r5, r10 = False, False, False

        embedding_times.append(embedding_time)
        searching_times.append(searching_time)
        total_times.append(total_time)

        # R@1 계산
        if predicted_segments[0]['Segment_name'] == true_segment_name:
            correct_count_r1 += 1
            r1 = True
            print('R@1 hit!')
        if predicted_segments[0]['Segment_name'][16:27] == true_segment_name[16:27]:
            correct_video_count_r1 += 1
            print('R@1-Video hit!')

        # R@5 계산
        top_5_results = predicted_segments[:5]
        for hit in top_5_results:
            if hit['Segment_name'] == true_segment_name:
                correct_count_r5 += 1
                r5 = True
                print('R@5 hit!')
            if hit['Segment_name'][16:27] == true_segment_name[16:27]:
                correct_video_count_r5 += 1
                print('R@5-Video hit!')
                break

        # R@10 계산
        top_10_results = predicted_segments[:10]
        for hit in top_10_results:
            if hit['Segment_name'] == true_segment_name:
                correct_count_r10 += 1
                r10 = True
                print('R@10 hit!')
            if hit['Segment_name'][16:27] == true_segment_name[16:27]:
                correct_video_count_r10 += 1
                print('R@10-Video hit!')
                break

        # 새로운 행 추가
        new_row = pd.DataFrame({
            'Query_Type': query_type, 
            'Query': query, 
            'Prediction': ", ".join([x['Segment_name'] for x in predicted_segments[:10]]), 
            'GT': true_segment_name, 
            'R@1': 1 if r1 else 0, 
            'R@5': 1 if r5 else 0, 
            'R@10': 1 if r10 else 0, 
            'embedding_time': embedding_time, 
            'searching_time': searching_time, 
            'total_time': total_time
        }, index=[0])

        evaluations = pd.concat([evaluations, new_row], ignore_index=True)  # ignore_index=True 추가

    # R@1, R@5, R@10 계산
    r1 = correct_count_r1 / total_queries
    r5 = correct_count_r5 / total_queries
    r10 = correct_count_r10 / total_queries
    r1_video = correct_video_count_r1 / total_queries
    r5_video = correct_video_count_r5 / total_queries
    r10_video = correct_video_count_r10 / total_queries

    # 시간 관련 평균 및 표준편차 계산
    mean_embedding_time = np.mean(embedding_times)
    std_embedding_time = np.std(embedding_times)
    
    mean_searching_time = np.mean(searching_times)
    std_searching_time = np.std(searching_times)
    
    mean_total_time = np.mean(total_times)
    std_total_time = np.std(total_times)

    # 최종 평가 점수 행 추가
    final_scores = pd.DataFrame({
        'Query_Type': ["Final Score"], 
        'Query': [""], 
        'Prediction': [""], 
        'GT': [""], 
        'R@1': [r1], 
        'R@5': [r5], 
        'R@10': [r10], 
        'embedding_time': [f"{mean_embedding_time:.4f} ± {std_embedding_time:.4f}"], 
        'searching_time': [f"{mean_searching_time:.4f} ± {std_searching_time:.4f}"], 
        'total_time': [f"{mean_total_time:.4f} ± {std_total_time:.4f}"]
    })

    evaluations = pd.concat([evaluations, final_scores], ignore_index=True)

    return r1, r5, r10, r1_video, r5_video, r10_video, evaluations

if __name__ == "__main__":
    import sys
    csv_path = '../test.csv'
    r1, r5, r10, r1_video, r5_video, r10_video, evaluations = calculate_r1_r5_r10(csv_path)

    print(f"R@1: {r1:.4f}, by video {r1_video:.4f}")
    print(f"R@5: {r5:.4f}, by video {r5_video:.4f}")
    print(f"R@10: {r10:.4f}, by video {r10_video:.4f}")

    # 최종 결과를 CSV 파일로 저장
    evaluations.to_csv("evaluation_results_CLIP_new.csv", index=False)
    print("Results saved to evaluation_results_CLIP_new.csv")