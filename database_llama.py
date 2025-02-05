import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model.sources.model_config import VideoChat2Config
from model.sources.modeling_videochat2 import InternVideo2_VideoChat2
import torch.nn.functional as F
from elasticsearch import Elasticsearch
import pandas as pd
from translate import translation
import math
from bert_score import score

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
    def __init__(self, model_path):
        # 설정 로드
        self.config = VideoChat2Config.from_json_file(os.path.join('model/configs','config.json'))
        
        # 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_config.llm.pretrained_llm_path,
            trust_remote_code=True,
            use_fast=False,
            token=os.getenv('HF_TOKEN')
        )
        # 패딩 토큰 설정
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 초기화
        if torch.cuda.is_available():
            self.model = InternVideo2_VideoChat2.from_pretrained(
                model_path,
                config=self.config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).cuda()
        else:
            self.model = InternVideo2_VideoChat2.from_pretrained(
                model_path,
                config=self.config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        self.model.eval()

    def encode_text(self, text):
        """텍스트를 임베딩 벡터로 변환"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            text_embeds = self.model.lm.get_input_embeddings()(inputs['input_ids'])
            mean_embedding = text_embeds.mean(dim=1).squeeze()
            mean_embedding = mean_embedding.float().cpu().numpy()
            
            return mean_embedding

    def generate_embedding(self, caption):
        """embedding 생성"""
        embedding = self.encode_text(caption)
        return embedding

    def save_to_elasticsearch(self, client, segment_name, start_time, end_time, caption, caption_ko, embedding, index_name="video_embeddings"):
        """임베딩을 Elasticsearch에 저장"""
        doc = {
            'segment_name': segment_name,
            'start_time': start_time,
            'end_time': end_time,
            'caption': caption,
            'caption_ko': caption_ko,
            'caption_embedding': embedding.tolist()
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
    index_name = "video_embeddings"
    
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
                "caption_embedding": {"type": "dense_vector", "dims": 4096}
            }
        }
    }
    
    client.indices.create(index=index_name, body=mappings)
    print(f"Created new index: {index_name} with mappings")

def save_from_csv(csv_path: str):
    client = get_elasticsearch_client()
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model/weights')
    captioner = VideoCaption(model_path)
    
    for _, row in df.iterrows():
        # 이미 존재하는 segment_name인지 확인
        search_result = client.search(
            index="video_embeddings",
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
            # caption_ko가 없는 경우 caption_ko를 제외하고 입력
            if not math.isnan(row['caption_ko']):
                captioner.save_to_elasticsearch(
                    client=client,
                    segment_name=row['segment_name'],
                    start_time=row['start_time'],
                    end_time=row['end_time'],
                    caption=row['caption'],
                    caption_ko=row['caption_ko'],
                    embedding=embedding
                )
            else:
                captioner.save_to_elasticsearch(
                    client=client,
                    segment_name=row['segment_name'],
                    start_time=row['start_time'],
                    end_time=row['end_time'],
                    caption=row['caption'],
                    caption_ko=translation(row["caption"], typ='en'),
                    embedding=embedding
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
    client = get_elasticsearch_client()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model/weights')
    captioner = VideoCaption(model_path)
    
    # 쿼리 텍스트를 임베딩 벡터로 변환
    query_embedding = captioner.generate_embedding(query_text)
    
    # 벡터 유사도 검색 쿼리 구성
    search_query = {
        "size": 20,  # 상위 5개 결과 반환
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
        results = client.search(index="video_embeddings", body=search_query)
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
    except Exception as e:
        print(f"Error during search: {str(e)}")

def prefiltering(query_text: str):
    client = get_elasticsearch_client()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model/weights')
    captioner = VideoCaption(model_path)
    
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
        results = client.search(index="video_embeddings", body=search_query)
        hits = results['hits']['hits']
        return hits
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return []

def two_stage_search_videos(query_text: str):
    # Step 1: Elasticsearch를 사용한 초기 필터링
    hits = prefiltering(query_text)

    if not hits:
        print("No results found in Elasticsearch.")
        return []
    
    # Step 2: BERTScore를 사용하여 2차 필터링
    final_results = calculate_bertscore(query_text, hits)

    print(f"\n2 Stage Search Results")
    for result in final_results:
        print(f"\nSegment Name: {result['segment_name']}")
        print(f"Start Time: {result['start_time']}")
        print(f"End Time: {result['end_time']}")
        print(f"Caption (EN): {result['caption']}")
        print(f"Caption (KO): {result['caption_ko']}")
        print(f"Similarity Score (BERTScore): {result['bert_score']}")

    return final_results

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
        two_stage_search_videos(sys.argv[2])
    else:
        print("Invalid command or missing arguments")