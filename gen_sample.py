import json
import random

# 1. 원본 JSON 불러오기
with open("GSM8k_kor_refined2_QA_for_benchmark.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. id 359 제외
filtered_data = [item for item in data if item.get("id") != 359]

# 3. 랜덤으로 2000개 추출
sampled_data = random.sample(filtered_data, 2000)

# 4. 새로운 JSON 파일로 저장
with open("tasks.json", "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print("추출 완료! tasks.json 파일이 생성되었습니다.")