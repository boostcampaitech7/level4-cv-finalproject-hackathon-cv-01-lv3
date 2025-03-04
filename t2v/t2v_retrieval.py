from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from evaluation_clip_frames import search_videos_clip
from evaluation_snowflake import search_videos_snowflake, get_elasticsearch_client
from evaluation_snowflake import VideoCaption as Snowflake
from evaluation_clip_frames import VideoCaption as LongCLIP
from evaluation_snowflake import calculate_r1_r5_r10_single as evaluation_snowflake
import time
import numpy as np


def reciprocal_rank_fusion(results1, results2, k=60):
    """
    Rank Fusion (Reciprocal Rank Fusion) 방식으로 두 검색 결과를 병합.
    
    :param results1: 첫 번째 검색 결과 (score_mode="max")
    :param results2: 두 번째 검색 결과 (score_mode="avg")
    :param k: Rank에 적용할 조정 값 (기본값: 60)
    :return: 병합된 결과 리스트
    """
    fused_scores = defaultdict(float)

    # 첫 번째 결과 (score_mode="max") 처리
    for rank, (video_id, score) in enumerate(results1, start=1):
        fused_scores[video_id] += 1 / (rank + k)

    # 두 번째 결과 (score_mode="avg") 처리
    for rank, (video_id, score) in enumerate(results2, start=1):
        fused_scores[video_id] += 1 / (rank + k)

    # 점수 기준으로 정렬 (내림차순)
    fused_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    return fused_results

def search_videos_optimized(query_text: str, GT: str = None, evaluations: pd.DataFrame = None):
    start = time.time()
    print(f"start_time : {start}")
    client = get_elasticsearch_client()
    print(f"server started")

    
    stage1, stage2 = 0,0
    
    
    # 1차 검색: Snowflake 임베딩 기반 검색
    model = Snowflake()
    query_embedding = model.encode_text(query_text)
    
    search_query_1st = {
        "size": 100,  # Top 100개 가져오기
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'caption_embedding') + 1.0",
                    "params": {"query_vector": query_embedding.squeeze().tolist()}
                }
            }
        }
    }

    try:
        start_search_1st = time.time()
        results_1st = client.search(index="snowflake", body=search_query_1st)
        top_segments = [
            hit["_source"]["segment_name"]
            for hit in results_1st["hits"]["hits"]
        ]
        print(f"1 Stage Results, GT: {GT}")
        stage1 = 1 if top_segments[0] == GT else 0
        for x in top_segments:
            print(x)

        model2 = LongCLIP()
        query_embedding = model2.encode_text(query_text)

        # 2차 검색을 위한 데이터 필터링
        search_query_2nd = {
            "size": 10,  # 상위 10개 결과 반환
            "query": {
                "bool": {
                    "must": [
                        {
                            "nested": {
                                "path": "frame_embeddings",  # frame_embeddings 배열을 기준으로 검색
                                "query": {
                                    "script_score": {
                                        "query": {"match_all": {}},  # 모든 프레임에 대해 유사도 계산
                                        "script": {
                                            "source": "cosineSimilarity(params.query_vector, 'frame_embeddings.frame_embedding') + 1.0",
                                            "params": {"query_vector": query_embedding.squeeze().tolist()}
                                        }
                                    }
                                },
                                "inner_hits": {"size": 1},  # 유사도가 가장 높은 프레임을 반환
                                "score_mode": "avg"
                            }
                        },
                        {  # 1차 검색에서 나온 후보군 내에서만 검색
                            "terms": {
                                "segment_name": top_segments
                            }
                        }
                    ]
                }
            }
        }

        start_search_2nd = time.time()
        results_2nd = client.search(index="newnew_clip", body=search_query_2nd)
        top2_segments = [
            (hit["_source"]["segment_name"], hit['_score'])
            for hit in results_2nd["hits"]["hits"]
        ]
        print(f"2 Stage Results, GT: {GT}, {top2_segments[0][0]}")
        stage2 = 1 if top2_segments[0][0] == GT else 0
        for x in top2_segments:
            print(x)

        new_row = pd.DataFrame({'GT': [GT], "1 Stage R@1": 1 if stage1 else 0, "2 Stage R@1": 1 if stage2 else 0})
        print(new_row)
        evaluations = pd.concat([evaluations, new_row], ignore_index = True)

        return evaluations



            ###
    #     max_score = -1
    #     best_segment_name = None

    #     # 2차 검색: frame_embeddings에서 가장 높은 유사도를 가지는 segment 찾기
    #     for hit in results_2nd["hits"]["hits"]:
    #         source = hit["_source"]
    #         for frame in source["frame_embeddings"]:
    #             frame_embedding = np.array(frame["frame_embedding"])
    #             similarity = np.dot(query_embedding, frame_embedding)  # 코사인 유사도 계산
    #             if similarity > max_score:
    #                 max_score = similarity
    #                 best_segment_name = source["segment_name"]

    #     end = time.time()

    #     # 실행 시간 측정
    #     embedding_time = start_search_1st - start
    #     searching_time_1st = start_search_2nd - start_search_1st
    #     searching_time_2nd = end - start_search_2nd
    #     total_time = end - start

    #     print(f"Embedding: {embedding_time:.3f} seconds!")
    #     print(f"1st Searching: {searching_time_1st:.3f} seconds!")
    #     print(f"2nd Searching: {searching_time_2nd:.3f} seconds!")
    #     print(f"Total: {total_time:.3f} seconds!")

    #     if best_segment_name:
    #         print(f"Best Segment Name: {best_segment_name}")
    #         print(f"Highest Similarity Score: {max_score}")

    #     return [{"Segment_name": best_segment_name, "Similarity Score": max_score}], embedding_time, searching_time_1st + searching_time_2nd, total_time

    except Exception as e:
        print(f"Error during search: {str(e)}")
        return [], 0, 0, 0



def calculate_r1_r5_r10(csv_path):
    df = pd.read_csv(csv_path)
    evaluations = pd.DataFrame(columns=['Query_Type', 'Query', 'Prediction', 'GT', 'R@1', 'R@5', 'R@10', 'embedding_time', 'searching_time', 'total_time'])
    df = df.iloc[:10,:] # 임시
    correct_count_r1 = 0
    correct_count_r5 = 0
    correct_count_r10 = 0
    correct_video_count_r1 = 0
    correct_video_count_r5 = 0
    correct_video_count_r10 = 0
    total_queries = len(df)

    embedding_times = []
    searching_times = []
    total_times = []

    for _, row in tqdm(df.iterrows(), total=total_queries, desc="Calculating R@1, R@5, R@10", unit="query"):
        query = row["Generated_Query"]
        query_type = row['Query_Type'] if len(row['Query_Type']) else ""
        true_segment_name = row["segment_name"]  # 실제 정답 segment_name
        print(f"GT: {true_segment_name}") # TODO 확인용. 나중에 지우기
        

        # TODO 나중에 embedding_time, searching_time, total_time 지우기
        predicted_segments_clip, embedding_time, searching_time, total_time = search_videos_clip(query)
        predicted_segments_snowflake, embedding_time, searching_time, total_time = search_videos_snowflake(query)
        ###
        #여기서 reciprocal_rank_fusion 수행 
        # 
        ###  
        r1, r5, r10 = False, False, False

        embedding_times.append(embedding_time)
        searching_times.append(searching_time)
        total_times.append(total_time)

        # R@1 계산
        if predicted_segments[0]['Segment_name'] == true_segment_name:
            correct_count_r1 += 1
            r1 = True
            print('R@1 hit!')
        if predicted_segments[0]['Segment_name'][16:27] == true_segment_name[16:27]:
            correct_video_count_r1 += 1
            print('R@1-Video hit!')

        # R@5 계산
        top_5_results = predicted_segments[:5]
        for hit in top_5_results:
            if hit['Segment_name'] == true_segment_name:
                correct_count_r5 += 1
                r5 = True
                print('R@5 hit!')
            if hit['Segment_name'][16:27] == true_segment_name[16:27]:
                correct_video_count_r5 += 1
                print('R@5-Video hit!')
                break

        # R@10 계산
        top_10_results = predicted_segments[:10]
        for hit in top_10_results:
            if hit['Segment_name'] == true_segment_name:
                correct_count_r10 += 1
                r10 = True
                print('R@10 hit!')
            if hit['Segment_name'][16:27] == true_segment_name[16:27]:
                correct_video_count_r10 += 1
                print('R@10-Video hit!')
                break

        # 새로운 행 추가
        new_row = pd.DataFrame({
            'Query_Type': query_type, 
            'Query': query, 
            'Prediction': ", ".join([x['Segment_name'] for x in predicted_segments[:10]]), 
            'GT': true_segment_name, 
            'R@1': 1 if r1 else 0, 
            'R@5': 1 if r5 else 0, 
            'R@10': 1 if r10 else 0, 
            'embedding_time': embedding_time, 
            'searching_time': searching_time, 
            'total_time': total_time
        }, index=[0])

        evaluations = pd.concat([evaluations, new_row], ignore_index=True)  # ignore_index=True 추가

    # R@1, R@5, R@10 계산
    r1 = correct_count_r1 / total_queries
    r5 = correct_count_r5 / total_queries
    r10 = correct_count_r10 / total_queries
    r1_video = correct_video_count_r1 / total_queries
    r5_video = correct_video_count_r5 / total_queries
    r10_video = correct_video_count_r10 / total_queries

    # 시간 관련 평균 및 표준편차 계산
    mean_embedding_time = np.mean(embedding_times)
    std_embedding_time = np.std(embedding_times)
    
    mean_searching_time = np.mean(searching_times)
    std_searching_time = np.std(searching_times)
    
    mean_total_time = np.mean(total_times)
    std_total_time = np.std(total_times)

    # 최종 평가 점수 행 추가
    final_scores = pd.DataFrame({
        'Query_Type': ["Final Score"], 
        'Query': [""], 
        'Prediction': [""], 
        'GT': [""], 
        'R@1': [r1], 
        'R@5': [r5], 
        'R@10': [r10], 
        'embedding_time': [f"{mean_embedding_time:.4f} ± {std_embedding_time:.4f}"], 
        'searching_time': [f"{mean_searching_time:.4f} ± {std_searching_time:.4f}"], 
        'total_time': [f"{mean_total_time:.4f} ± {std_total_time:.4f}"]
    })

    evaluations = pd.concat([evaluations, final_scores], ignore_index=True)

    return r1, r5, r10, r1_video, r5_video, r10_video, evaluations


if __name__ == '__main__':
    csv_path = '../test.csv'
    df = pd.read_csv(csv_path)
    evaluations = pd.DataFrame(columns=['GT', "1 Stage R@1", "2 Stage R@1"])
    for idx, i in enumerate(range(len(df))):
        row = df.iloc[i,:]
        evaluations = search_videos_optimized(row['Generated_Query'], row['segment_name'], evaluations)
        # print(f"row: {row}")
        # print(type(row))
        # result = evaluation_snowflake(row)
    total_1stage = evaluations['1 Stage R@1'].sum() / len(evaluations)
    total_2stage = evaluations['2 Stage R@1'].sum() / len(evaluations)
    final_row = pd.DataFrame({'GT': ['Final Result'], '1 Stage R@1' : total_1stage, '2 Stage R@1' : total_2stage})
    evaluations = pd.concat([evaluations, final_row], ignore_index=True)
    evaluations.to_csv('Evaluations_Fusion.csv', index=False)