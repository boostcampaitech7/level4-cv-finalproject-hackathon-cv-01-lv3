import json

# 텍스트 파일 읽기
with open('test_list_new.txt', 'r') as f:
    video_ids = [line.strip() for line in f.readlines()]

# JSON 형식으로 변환
json_data = []
for video_id in video_ids:
    json_data.append({
        "video_id": video_id,
        "caption": ""  # 캡션이 필요한 경우 여기에 추가
    })

# JSON 파일로 저장
with open('test_data.json', 'w') as f:
    for item in json_data:
        f.write(json.dumps(item) + '\n')