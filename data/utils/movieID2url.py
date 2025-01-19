import requests
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_youtube_id(yt8m_id, session):
    """
    YT8M 데이터셋의 4자리 ID를 실제 YouTube 비디오 ID로 변환
    
    Args:
        yt8m_id (str): 4자리 YT8M ID (예: 'nXSc')
        session: requests 세션 객체
    
    Returns:
        str: YouTube 비디오 ID
    """
    # URL 구성
    prefix = yt8m_id[:2]  # 처음 2글자
    base_url = f"http://data.yt8m.org/2/j/i/{prefix}/{yt8m_id}.js"
    
    # 브라우저처럼 보이는 헤더 추가
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/javascript,application/javascript,application/ecmascript,application/x-ecmascript,*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive'
    }
    
    try:
        # 데이터 가져오기 (1초 대기)
        time.sleep(1)  # 딜레이 증가
        
        response = session.get(base_url, headers=headers)
        response.raise_for_status()
        
        # 응답 텍스트 파싱 (예: i("nXSc","0sf943sWZls");)
        text = response.text.strip()
        
        # 따옴표로 둘러싸인 두 번째 값(YouTube ID) 추출
        parts = text.split('"')
        if len(parts) >= 4:
            youtube_id = parts[3]  # 두 번째 따옴표 쌍 안의 값
            return youtube_id
            
        raise ValueError(f"Unexpected response format: {text}")
        
    except Exception as e:
        print(f"Error fetching YouTube ID for {yt8m_id}: {str(e)}")
        return None

def create_session():
    """재시도 로직이 포함된 세션 생성"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # 최대 재시도 횟수
        backoff_factor=2,  # 재시도 간 대기 시간 증가 계수
        status_forcelist=[403, 429, 500, 502, 503, 504]  # 재시도할 HTTP 상태 코드
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def convert_ids(input_file, output_file):
    """
    YT8M ID를 YouTube ID로 변환하고 파일에 저장
    
    Args:
        input_file (str): YT8M ID가 저장된 입력 파일 경로
        output_file (str): YouTube ID를 저장할 출력 파일 경로
    """
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 세션 생성
    session = create_session()
    
    # 파일에서 ID 읽기
    yt8m_ids = []
    with open(input_file, "r") as f:
        for line in f:
            yt8m_ids.append(line.strip())
    
    # 중복 제거
    yt8m_ids = list(set(yt8m_ids))
    print(f"총 {len(yt8m_ids)}개의 고유한 ID를 처리합니다.")
    
    # 결과 저장
    with open(output_file, "w") as f:
        for i, yt8m_id in enumerate(yt8m_ids, 1):
            youtube_id = get_youtube_id(yt8m_id, session)
            if youtube_id:
                f.write(f"{yt8m_id}\t{youtube_id}\n")
                
            # 진행 상황 출력
            if i % 10 == 0:
                print(f"처리 중... {i}/{len(yt8m_ids)} ({(i/len(yt8m_ids)*100):.1f}%)")

if __name__ == "__main__":
    # 입력/출력 파일 경로 설정
    input_file = "./data/yt8m/combined_found_videos.txt"
    output_file = "./data/yt8m/movie_clips_youtube_ids.txt"
    
    convert_ids(input_file, output_file)