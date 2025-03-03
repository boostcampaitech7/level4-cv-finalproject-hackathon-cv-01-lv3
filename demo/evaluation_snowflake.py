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
from sentence_transformers import SentenceTransformer
import numpy as np
# main()


INDEX_NAME = "snowflake"
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
        # snowflake-arctic-embed-l-v2.0 모델 로드
        self.model = SentenceTransformer("./weights_snowflake-arctic-embed-l-v2.0", device=DEVICE)
        self.model.eval()  # 추론 모드

    def encode_text(self, text):
        """텍스트를 임베딩 벡터로 변환"""
        with torch.no_grad():
            text_embedding = self.model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        return text_embedding.cpu().numpy()

    def generate_embedding(self, caption):
        """embedding 생성"""
        embedding = self.encode_text(caption)
        return embedding
    
    def save_to_elasticsearch(self, client, segment_name, start_time, end_time, caption, caption_ko, embedding, index_name=INDEX_NAME):
        """임베딩을 Elasticsearch에 저장"""
        doc = {
            'segment_name': segment_name,
            'start_time': start_time,   
            'end_time': end_time,
            'caption': caption,
            'caption_ko': caption_ko,
            'caption_embedding': embedding.tolist(),
        }
        
        try:
            response = client.index(index=index_name, document=doc)
            print(f"Document indexed: {response['result']}")
            print(f"Embedding dimension: {len(doc['caption_embedding'])}")
        except Exception as e:
            print(f"Error saving to Elasticsearch: {str(e)}")
            print(f"Embedding shape: {embedding.shape}")

model = VideoCaption()

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
                "caption_embedding": {"type": "dense_vector", "dims": 1024},
            }
        }
    }
    
    client.indices.create(index=index_name, body=mappings)
    print(f"Created new index: {index_name} with mappings")

def save_from_csv(csv_path: str):
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
        
        try:
            # 임베딩 생성
            embedding = captioner.generate_embedding(row['caption'])
            # Elasticsearch에 저장
            # caption_ko를 제외하고 입력
            captioner.save_to_elasticsearch(
                client=client,
                segment_name=row['segment_name'],
                start_time=row['start_time'],
                end_time=row['end_time'],
                caption=row['caption'],
                caption_ko="",
                embedding=embedding,
                index_name=INDEX_NAME
            )
            print(f"Successfully processed: {row['segment_name']}")
            
        except Exception as e:
            print(f"Error processing {row['segment_name']}: {str(e)}")
            continue

def calculate_bertscore(query_text: str, hits):
    # BERTScore 계산
    candidate_captions = [hit['_source']['caption'] for hit in hits]

    P, R, F1 = score([query_text] * len(candidate_captions), candidate_captions, lang='en')

    # 각 비디오에 대해 BERTScore F1 점수와 함께 반환
    scored_results = []
    for i, hit in enumerate(hits):
        hit_score = F1[i].item()
        hit_data = hit['_source']
        hit_data['bert_score'] = hit_score
        scored_results.append(hit_data)
    
    scored_results = sorted(scored_results, key=lambda x: x['bert_score'], reverse=True)
    
    return scored_results[:5]

def search_videos(query_text: str):
    start = time.time()
    print(f"start_time : {start}")
    client = get_elasticsearch_client()
    print(f"server started")
    # 쿼리 텍스트를 임베딩 벡터로 변환
    global model
    query_embedding = model.generate_embedding(query_text)
    
    # 벡터 유사도 검색 쿼리 구성
    search_query = {
        "size": 10,  # 상위 5개 결과 반환
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'caption_embedding') + 1.0",
                    "params": {"query_vector": query_embedding.tolist()}
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
        for hit in results['hits']['hits']:
            score = hit['_score']
            source = hit['_source']
            print(f"\nSegment Name: {source['segment_name']}")
            print(f"Start Time: {source['start_time']}")
            print(f"End Time: {source['end_time']}")
            print(f"Caption (EN): {source['caption']}")
            print(f"Caption (KO): {source['caption_ko']}")
            print(f"Similarity Score: {score}")
            retrieve.append({'Segment_name': source['segment_name'], 'Start_time': source['start_time'], 'End_time': source['end_time'], 'Similarity Score': score})
        end = time.time()
        embedding_time = start_search - start
        searching_time = end - start_search
        total_time = end - start
        print(f"Embedding: {embedding_time: .3f}seconds!")
        print(f"Searching: {searching_time: .3f}seconds!")
        print(f"Embedding + Searching: {total_time: .3f}seconds!")
    except Exception as e:
        print(f"Error during search: {str(e)}")
    return retrieve, embedding_time, searching_time, total_time

def prefiltering(query_text: str):
    client = get_elasticsearch_client()
    captioner = VideoCaption()
    
    # 쿼리 텍스트를 임베딩 벡터로 변환
    query_embedding = captioner.generate_embedding(query_text)
    
    # 벡터 유사도 검색 쿼리 구성
    search_query = {
        "size": 20,  # 상위 20개 결과 반환
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'caption_embedding') + 1.0",
                    "params": {"query_vector": query_embedding.tolist()}
                }
            }
        }
    }
    
    # 검색 실행
    try:
        results = client.search(index=INDEX_NAME, body=search_query)
        hits = results['hits']['hits']
        return hits
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return []


def run(query_text: str)-> tuple[str, str]:
    results, segment_name = search_videos(query_text)

    print(f"\n\n[DEBUG] results: {results}")
    print(f"[DEBUG] segment_name: {segment_name}")
    return results, segment_name

def find_video(data_path: str, segment_name: str):  
    dsrc, category, _, _ = segment_name.split('_')
    video_path = os.path.join(data_path, dsrc.upper(), category)
    for x in ['train', 'test']:
        if os.path.exists(os.path.join(video_path, x, 'clips', f'{segment_name}.mp4')):
            return os.path.join(video_path, x, 'clips', f'{segment_name}.mp4')
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
    evaluations.to_csv("evaluation_results_snowflake.csv", index=False)
    print("Results saved to evaluation_results_snowflake.csv")