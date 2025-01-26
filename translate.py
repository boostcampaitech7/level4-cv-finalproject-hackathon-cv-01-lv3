import httpx
import asyncio
from googletrans import Translator

async def translation(caption: str, typ: str) -> str:
    """
    번역을 수행하는 함수
    주의 사항: 이 함수는 asynchronous하게 작동합니다
    번역 실패 시 원본 문장 반환
    --------------------
    args
    caption: 번역할 문장 또는 단어
    typ: 번역할 문장의 언어 (한국어로 입력 받을 시 ko, 영어로 입력 받을 시 en으로 설정)

    출력: 번역된 문장 또는 단어 (str)
    """
    translator = Translator(timeout=30)
    if typ == 'ko':
        src, dest = 'ko', 'en'
    if typ == 'en':
        src, dest = 'en', 'ko'
    retries = 5
    for attempt in range(retries):
        try:
            res = await translator.translate(caption, src=src, dest=dest)
            return res.text
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            if attempt == retries - 1:
                print(f"Translation failed after {retries} attempts")
                return caption
            print(f"Translation timeout (attempt {attempt + 1}/{retries}). Retrying...")
            await asyncio.sleep(2 * (attempt + 1))