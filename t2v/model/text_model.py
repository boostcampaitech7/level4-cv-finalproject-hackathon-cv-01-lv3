import os
import torch
from sentence_transformers import SentenceTransformer
import torch
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # level4 폴더 기준
weights_path = os.path.join(base_dir, "weights")
sys.path.append(weights_path)  # weights 폴더를 모듈 경로에 추가

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Snowflake:
    """
    Text-Text 간 Similarity를 비교할 수 있는 Snowflake 모델을 가용할 수 있는 클래스
    """
    def __init__(self):
        # snowflake-arctic-embed-l-v2.0 모델 로드
        global base_dir  # level4 폴더 기준
        weights_path = os.path.join(base_dir, "weights", "weights_snowflake-arctic-embed-l-v2.0")
        self.model = SentenceTransformer(weights_path, device=DEVICE)
        self.model.eval()  # 추론 모드

    def encode_text(self, text: str):
        """텍스트를 임베딩 벡터로 변환"""
        with torch.no_grad():
            text_embedding = self.model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        return text_embedding

class MiniLM:
    def __init__(self):
        # 설정 로드
        # SBERT 기반 모델 로드
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # 토크나이저 초기화

    def encode_text(self, text):
        """텍스트를 임베딩 벡터로 변환"""
        with torch.no_grad():
            text_embedding = self.model.cuda().encode(text, normalize_embeddings=True)
        return text_embedding