import os
import pandas as pd
from elasticsearch import Elasticsearch

INDEX_NAME = "movieclips"

def get_elasticsearch_client():
    elasticsearch_url = os.environ.get('ELASTICSEARCH_URL')
    elasticsearch_api_key = os.environ.get('ELASTICSEARCH_API_KEY')

    if not elasticsearch_url or not elasticsearch_api_key:
        raise ValueError("환경 변수 ELASTICSEARCH_URL과 ELASTICSEARCH_API_KEY를 설정해주세요.")
    
    return Elasticsearch(
        elasticsearch_url,
        api_key=elasticsearch_api_key
    )

def fetch_all_documents(index_name, size=5000):
    client = get_elasticsearch_client()
    
    query = {
        "size": size,
        "query": {"match_all": {}}
    }
    
    response = client.search(index=index_name, body=query, scroll='2m')
    
    scroll_id = response['_scroll_id']
    hits = response['hits']['hits']
    
    all_hits = []
    while len(hits):
        all_hits.extend(hits)
        response = client.scroll(scroll_id=scroll_id, scroll='2m')
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
    
    client.clear_scroll(scroll_id=scroll_id)
    
    return [hit['_source'] for hit in all_hits]

def save_to_csv(index_name, output_file):
    data = fetch_all_documents(index_name)
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Saved {len(df)} records to {output_file}")

if __name__ == "__main__":
    save_to_csv(INDEX_NAME, "elasticsearch_data.csv")
