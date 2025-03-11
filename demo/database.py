import os
import sys
from elasticsearch import Elasticsearch
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from t2v.evaluation_snowflake import Snowflake
from t2v.evaluation_longclip import LongCLIP

# level4-cv-finalproject-hackathon-cv-01-lv3/t2v 폴더가 필요합니다.
# level4-cv-finalproject-hackathon-cv-01-lv3/t2v/weights 내부에 snowflake, longclip에 대한 weight가 저장되어 있어야 합니다.

# TT: Text-Text Similarity 계산, FT: Frame(Video)-Text Similarity 계산 
TT_INDEX_NAME = "text_text"
FT_INDEX_NAME = "frame_text"

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
        # Text-Text Similarity 계산용 임베딩 모델 (ttmodel), Frame-Text Similarity 계산용 임베딩 모델 (ftmodel) 초기화
        self.ttmodel = Snowflake()
        self.ftmodel = LongCLIP()

    def _tt_encode_text(self, text: str):
        """Text-to-Text 비교를 위해 텍스트를 임베딩 벡터로 변환"""
        tt_text_embedding = self.ttmodel.encode_text(text)
        return tt_text_embedding
        
    def _ft_encode_text(self, text: str):
        """Video (Frame)-to-Text 비교를 위해 텍스트를 임베딩 벡터로 변환"""
        ft_text_embedding = self.ftmodel.encode_text(text)
        return ft_text_embedding
    
    def _ft_encode_video(self, video_path: str, num_frames: int=8):
        """Video (Frame)-to-Text 비교를 위해 비디오를 8장의 프레임 임베딩 벡터로 변환
        num_frames argument를 수정하여 몇 개의 프레임을 Vector DB에 저장할 지 조절할 수 있습니다."""
        ft_video_embedding = self.ftmodel.encode_video(video_path, num_frames)
        return ft_video_embedding

    def generate_embedding_search(self, textquery: str):
        """검색을 위해 Input Text Query를 Text 임베딩 벡터로 변환 및 생성"""
        tt_text_embedding = self._tt_encode_text(textquery)
        ft_text_embedding = self._ft_encode_text(textquery)
        return tt_text_embedding, ft_text_embedding
    
    def generate_embedding_save(self, caption: str, video_path: str):
        """Vector DB에 데이터를 넣기 위해 Caption과 Video를 받아 임베딩 벡터로 변환 후, 반환"""
        tt_text_embedding = self._tt_encode_text(caption)
        ft_video_embedding = self._ft_encode_video(video_path)
        return tt_text_embedding, ft_video_embedding

    def save_to_elasticsearch(self, client, row: pd.Series, data_path=None, TT_index_name=TT_INDEX_NAME, FT_index_name=FT_INDEX_NAME):
        """임베딩을 각각의 Elasticsearch에 저장"""
        video_path = get_video(video_name=row['segment_name'], data_path=data_path)
        if video_path is None:
            print(f"Failed to save {row['segment_name']}. The video is missing!")
            return None
        
        # 임베딩 생성
        tt_text_embedding, ft_video_embedding = model.generate_embedding_save(row['caption'], video_path)
        frame_embeddings_list = [{"frame_index": emb['frame_index'], "frame_embedding": emb['frame_embedding'].squeeze().tolist()} for emb in ft_video_embedding]

        # 1. Text-to-Text Similarity 계산용 VectorDB 작업
        # 이미 존재하는 segment_name인지 확인
        search_result_1st = client.search(
            index=TT_INDEX_NAME,
            body={
                "query": {
                    "term": {
                        "segment_name": row['segment_name']
                    }
                }
            }
        )
        
        # 이미 존재하는 경우 건너뛰기
        if search_result_1st['hits']['total']['value'] > 0:
            print(f"Skipping {row['segment_name']}: Already exists")
        else:
            tt_doc = {
                'segment_name': row['segment_name'],
                'start_time': row['start_time'],   
                'end_time': row['end_time'],
                'caption': row['caption'],
                'caption_ko': row['caption_ko'] if not pd.isna(row['caption_ko']) else "",
                'caption_embedding': tt_text_embedding.tolist()
            }
            try:
                tt_response = client.index(index=TT_index_name, document=tt_doc)
                print(f"Text-to-Text Document indexed: {tt_response['result']}")
                print(f"Text-to-Text Embedding dimension: {len(tt_doc['caption_embedding'])}")
            except Exception as e:
                print(f"Error saving to Elasticsearch: {str(e)}")
                print(f"Text-to-Text Caption_Embedding shape: {tt_text_embedding.shape}")
        
        # 2. Frame-to-Text Similarity 계산용 VectorDB 작업
        # 이미 존재하는 segment_name인지 확인
        search_result_2nd = client.search(
            index=FT_INDEX_NAME,
            body={
                "query": {
                    "term": {
                        "segment_name": row['segment_name']
                    }
                }
            }
        )
        
        # 이미 존재하는 경우 건너뛰기
        if search_result_2nd['hits']['total']['value'] > 0:
            print(f"Skipping {row['segment_name']}: Already exists")
            video_path = get_video(video_name=row['segment_name'], data_path=data_path)

        if video_path is None:
            print(f"passing {row['segment_name']}")
        else:
            ft_doc = {
            'segment_name': row['segment_name'],
            'start_time': row['start_time'],   
            'end_time': row['end_time'],
            'caption': row['caption'],
            'caption_ko': row['caption_ko'] if not pd.isna(row['caption_ko']) else "",
            'frame_embeddings': frame_embeddings_list,  # 각 프레임의 임베딩 추가
        }
            try:
                ft_response = client.index(index=FT_index_name, document=ft_doc)
                print(f"Frame-to-Frame Document indexed: {ft_response['result']}")
                print(f"Frame-to-Frame Embedding dimension: {len(ft_doc['frame_embeddings'])}")
            except Exception as e:
                print(f"Error saving to Elasticsearch: {str(e)}")
                print(f"Text-to-Text Caption_Embedding shape: {tt_text_embedding.shape}")
                print(f"Frame-to-Text Frame_Embedding shape: {ft_video_embedding.shape}")

def create_index():
    """새로운 Vector DB Index를 만드는 함수"""
    client = get_elasticsearch_client()
    tt_index_name = TT_INDEX_NAME
    ft_index_name = FT_INDEX_NAME
    
    # 기존 인덱스가 있다면 삭제
    if client.indices.exists(index=tt_index_name):
        client.indices.delete(index=tt_index_name)
        print(f"Deleted existing index: {tt_index_name}")
    
    if client.indices.exists(index=ft_index_name):
        client.indices.delete(index=ft_index_name)
        print(f"Deleted existing index: {ft_index_name}")
    
    # 새로운 인덱스 생성 with mappings
    tt_mappings = {
        "mappings": {
            "properties": {
                "segment_name": {"type": "keyword"},
                "start_time": {"type": "keyword"},
                "end_time": {"type": "keyword"},
                "caption": {"type": "text"},
                "caption_ko": {"type": "text"},
                "caption_embedding": {"type": "dense_vector", "dims": 1024}
            }
        }
    }

    ft_mappings = {
        "mappings": {
            "properties": {
                "segment_name": {"type": "keyword"},
                "start_time": {"type": "keyword"},
                "end_time": {"type": "keyword"},
                "caption": {"type": "text"},
                "caption_ko": {"type": "text"},
                "frame_embeddings": {
                    "type": "nested",  # Nested 타입으로 설정
                    "properties": {
                        "frame_index": {"type": "integer"},
                        "frame_embedding": {"type": "dense_vector", "dims": 768}
                    }
                }
            }
        }
    }
    
    client.indices.create(index=tt_index_name, body=tt_mappings)
    client.indices.create(index=ft_index_name, body=ft_mappings)
    print(f"Created new index (For Text-to-Text Comparison): {tt_index_name} with mappings")
    print(f"Created new index (For Frame-to-Text Comparison): {ft_index_name} with mappings")

def save_from_csv(csv_path: str, data_path: str = '../../video'):
    """csv로부터 데이터를 불러와서 Vector DB에 저장하는 함수
    csv 필수 열: [segment_name, start_time, end_time, caption]
    csv 선택 열: [caption_ko] 등"""
    client = get_elasticsearch_client()
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    global model
    for _, row in df.iterrows():
        try:
            # Elasticsearch에 저장
            # caption_ko를 제외하고 입력
            model.save_to_elasticsearch(
                client=client,
                row=row,
                data_path=data_path,
                TT_index_name=TT_INDEX_NAME,
                FT_index_name=FT_INDEX_NAME
            )
            print(f"Successfully processed: {row['segment_name']}")
            
        except Exception as e:
            print(f"Error processing {row['segment_name']}: {str(e)}")
            continue


# 가중 Reciprocal Rank Fusion (Weighted RRF) 함수
def weighted_reciprocal_rank_fusion(results_list, weights: list, k=60):
    """RRF로 두가지의 결과를 합쳐서(Rank Fusion) 후 반환"""
    doc_rank_sum = {}
    for i, results in enumerate(results_list):
        weight = weights[i]
        for rank, doc_id in enumerate(results):
            if doc_id not in doc_rank_sum:
                doc_rank_sum[doc_id] = 0
            doc_rank_sum[doc_id] += weight * (1 / (rank + k))
    fused_results_weighted_rrf = sorted(doc_rank_sum.keys(), key=lambda doc_id: doc_rank_sum[doc_id], reverse=True)
    return fused_results_weighted_rrf

def calculate_2_stage_result(stage2_result: list):
    """Stage 2 (Frame-Text)결과를 Reranking하여 반환"""
    fin_results = []
    for seg in stage2_result:
        fin_results.append((seg[0], sum(seg[1:]) / len(seg[1:])))
    final_results = [x[0] for x in sorted(fin_results, reverse = True, key = lambda x: x[1])]
    return final_results

def search_videos(query_text: str, m: int=1, rrf_k: int = 60, weights: list=[0.5, 0.5]):
    """Vector DB에 Text Query를 넣어 가장 알맞는 데이터 반환"""
    client = get_elasticsearch_client()
    global model

    # 쿼리 텍스트를 임베딩 벡터로 변환
    query_embedding_tt = model.ttmodel.encode_text(query_text)
    query_embedding_ft = model.ftmodel.encode_text(query_text)
    

    # 벡터 유사도 검색 쿼리 구성
    search_query_1st = {
            "size": 10,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'caption_embedding') + 1.0",
                        "params": {"query_vector": query_embedding_tt.squeeze().tolist()}
                    }
                }
            }
        }
    try:
        # 1차 검색 (Text-to-Text 비교)
        results_1st = client.options(request_timeout=16000).search(index=TT_INDEX_NAME, body=search_query_1st)
        top_segments_stage1 = [
            hit["_source"]["segment_name"]
            for hit in results_1st["hits"]["hits"]
        ]

        name_caption_pair = dict()
        for hit in results_1st["hits"]["hits"]:
            name_caption_pair[hit["_source"]["segment_name"]] = dict()
            name_caption_pair[hit["_source"]["segment_name"]]['score'] = hit['_score']
            for info in ['start_time', 'end_time', 'caption', 'caption_ko']:
                name_caption_pair[hit["_source"]["segment_name"]][info] = hit['_source'][info]

    except Exception as e:
        print(f"Error during search: {str(e)}")

    search_query_2nd = {
            "size": 10,
            "query": {
                "bool": {
                    "must": [
                        {
                            "nested": {
                                "path": "frame_embeddings",
                                "query": {
                                    "script_score": {
                                        "query": {"match_all": {}},
                                        "script": {
                                            "source": "cosineSimilarity(params.query_vector, 'frame_embeddings.frame_embedding') + 1.0",
                                            "params": {"query_vector": query_embedding_ft.squeeze().tolist()}
                                        }
                                    }
                                },
                                "inner_hits": {"size": m, "sort": [{"_score": "desc"}]},  # top-m 프레임 선택
                                "score_mode": "avg"  # 평균 점수 사용
                            }
                        },
                        {
                            "terms": {
                                "segment_name": top_segments_stage1
                            }
                        }
                    ]
                }
            }
        }
    try:
        # 2차 검색 (Frame-to-Text 비교를 통한 Reranking)
        results_2nd = client.options(request_timeout=16000).search(index=FT_INDEX_NAME, body=search_query_2nd)
        inner_hits_stage2 = []
        for hit in results_2nd['hits']['hits']:
            inner = []
            seg_name = hit["_source"]["segment_name"]
            inner.append(seg_name)
            frames = hit["inner_hits"]["frame_embeddings"]["hits"]["hits"]
            for frame in frames:
                frame_score = frame["_score"]
                inner.append(frame_score)
            inner_hits_stage2.append(inner)
        top_segments_stage2 = calculate_2_stage_result(inner_hits_stage2)

        # Rank Fusion (RRF) 적용
        stage1_doc_ids = top_segments_stage1
        stage2_doc_ids = top_segments_stage2
        results_for_rrf = [stage1_doc_ids, stage2_doc_ids]
        fused_results_rrf = weighted_reciprocal_rank_fusion(results_for_rrf, weights=weights, k=rrf_k) # 가중 RRF 호출

        top_result = fused_results_rrf[0]

    except Exception as e:
        print(f"Error during search: {str(e)}")
    return f"Segment_name: {top_result}, Start_time: {name_caption_pair[top_result]['start_time']}, End_time: {name_caption_pair[top_result]['end_time']}, Caption: {name_caption_pair[top_result]['caption']}, Similarity Score: {name_caption_pair[top_result]['score']}", top_result


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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("1. Create index: python create_index.py create")
        print("2. Save from CSV: python create_index.py save <csv_path>")
        print("3. Search: python create_index.py search <query_text>")
        sys.exit(1)
    
    command = sys.argv[1]
    model = VideoCaption()
    if command == "create":
        create_index()
    elif command == "save" and len(sys.argv) == 3:
        save_from_csv(sys.argv[2])
    elif command == "search" and len(sys.argv) == 3:
        result = search_videos(sys.argv[2])
        print(result)
    else:
        print("Invalid command or missing arguments")

