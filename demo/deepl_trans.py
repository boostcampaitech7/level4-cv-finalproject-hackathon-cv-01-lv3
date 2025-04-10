import requests
from config import DeepLConfig  # 상대 경로 임포트  

def translate_deepl(text, source_lang="KO", target_lang="EN"):
    params = {
        "auth_key": DeepLConfig.API_KEY,
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang,
    }
    response = requests.post(DeepLConfig.URL, data=params)
    if response.status_code == 200:
        result = response.json()
        return result["translations"][0]["text"]
    else:
        raise Exception(f"DeepL API Error: {response.status_code} {response.text}")