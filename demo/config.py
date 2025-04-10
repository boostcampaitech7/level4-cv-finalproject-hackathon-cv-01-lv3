import os

class DeepLConfig:
    API_KEY = os.getenv("DEEPL_API_KEY", "YOUR_DEEPL_API_KEY")
    URL = os.getenv("DEEPL_API_URL", "https://api-free.deepl.com/v2/translate")

class ServiceConfig:
    # STT 서비스 설정 (별도 서버 필요)
    STT_URL = os.getenv("STT_SERVICE_URL", "YOUR_STT_SERVICE_URL/Endpoint")
    # 요약 서버 설정(별도 서버 필요)
    SUMMARY_URL = os.getenv("SUMMARY_SERVICE_URL", "YOUR_SUMMARY_SERVICE_URL/Endpoint")
    