import os
import torch
from elasticsearch import Elasticsearch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm
from torch import Tensor
from download import main
from torch.cuda.amp import autocast
import numpy as np
import onnxruntime as ort

# main()


INDEX_NAME = "lens-d4000"
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
        # LENS-d4000 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained("./weights_yibinlei/LENS-d4000")
        self.model = AutoModel.from_pretrained("./weights_yibinlei/LENS-d4000").half().to(DEVICE, dtype=torch.bfloat16)
        self.model.eval()  # 추론 모드
        print(f"model created")
        
    def encode_text(self, text):
        """텍스트를 임베딩 벡터로 변환"""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            with torch.amp.autocast(DEVICE):
                # 모델 실행
                outputs = self.model(**inputs)
            # LENS-d8000 모델의 경우 마지막 hidden state를 사용
            text_embedding = self.pooling_func(outputs.last_hidden_state, inputs['attention_mask'])  # 평균을 사용해 벡터화
            
            return text_embedding.squeeze(0).cpu().numpy()
    
    def encode_text_sbert(self, text):
        with torch.no_grad():
            text_embedding = self.model2.cuda().encode(text, normalize_embeddings=True)
        return text_embedding
    
    def pooling_func(self, vecs: Tensor, pooling_mask: Tensor) -> Tensor:
        # max-pooling을 사용
        return torch.max(torch.log(1 + torch.relu(vecs)) * pooling_mask.unsqueeze(-1), dim=1).values

    def generate_embedding(self, caption):
        """embedding 생성"""
        embedding = self.encode_text(caption)
        return embedding
    
    def save_to_elasticsearch(self, client, segment_name, start_time, end_time, caption, caption_ko, embedding, embedding_sbert, index_name=INDEX_NAME):
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
                "caption_embedding": {"type": "dense_vector", "dims": 4096},
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
    model = VideoCaption()
    query_embedding = model.generate_embedding(query_text)
    
    # 벡터 유사도 검색 쿼리 구성
    search_query = {
        "size": 1,  # 상위 5개 결과 반환
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
        end = time.time()
        print(f"Embedding: {start_search - start: .3f}seconds!")
        print(f"Searching: {end - start_search: .3f}seconds!")
        print(f"Embedding + Searching: {end - start: .3f}seconds!")
    except Exception as e:
        print(f"Error during search: {str(e)}")
    return f"Segment_name: {source['segment_name']}, Start_time: {source['start_time']}, End_time: {source['end_time']}, Similarity Score: {score}", source['segment_name']

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

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("1. Create index: python create_index.py create")
        print("2. Save from CSV: python create_index.py save <csv_path>")
        print("3. Search: python create_index.py search <query_text>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create":
        create_index()
    elif command == "save" and len(sys.argv) == 3:
        save_from_csv(sys.argv[2])
    elif command == "search" and len(sys.argv) == 3:
        search_videos(sys.argv[2])
    else:
        print("Invalid command or missing arguments")

