import os
import torch
from sentence_transformers import SentenceTransformer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class VideoCaption:
    """
    Text-Text 간 Similarity를 비교할 수 있는 Snowflake 모델을 가용할 수 있는 클래스
    """
    def __init__(self):
        # snowflake-arctic-embed-l-v2.0 모델 로드
        base_dir = os.path.dirname(os.path.abspath(__file__))  # level4 폴더 기준
        weights_path = os.path.join(base_dir, "weights", "weights_snowflake-arctic-embed-l-v2.0")
        self.model = SentenceTransformer(weights_path, device=DEVICE)
        self.model.eval()  # 추론 모드

    def encode_text(self, text: str):
        """텍스트를 임베딩 벡터로 변환"""
        with torch.no_grad():
            text_embedding = self.model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        return text_embedding.cpu().numpy()