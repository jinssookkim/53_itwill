import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드
df = pd.read_csv('final.csv')

# 2. 기간 필터링 (2024년 1분기 ~ 2025년 3분기)
filtered_df = df[
    ((df['연도'] == 2024)) | 
    ((df['연도'] == 2025) & (df['분기'] <= 3))
].copy()

# 3. 분석에 사용할 컬럼 설정
# - IS_WORKER_MISSING, IS_RENT_MISSING 추가
# - 철도_역_수 제외
selected_cols = [
    '유사_업종_점포_수', '총_유동인구_수', '총_직장_인구_수', 
    'IS_WORKER_MISSING',  # 직장인구 결측 여부 추가
    '집객시설_수', '관공서_수', '은행_수', '종합병원_수', '일반_병원_수', 
    '약국_수', '유치원_수', '초등학교_수', '중학교_수', '고등학교_수', 
    '대학교_수', '백화점_수', '슈퍼마켓_수', '극장_수', '숙박_시설_수', 
    '공항_수', '버스_터미널_수', '지하철_역_수',  # 철도_역_수 삭제
    '버스_정거장_수', '추정매출액', 
    'IS_RENT_MISSING',    # 임대시세 결측 여부 추가
    '전체_임대시세', '영역_면적', '지출총금액'
]

# 4. 결측치 0으로 채우기 및 전처리된 데이터셋 생성
data_cleaned = filtered_df[selected_cols].fillna(0)

# (선택) 클러스터링용으로 만든 데이터셋을 따로 저장해두고 싶다면 아래 주석을 해제하세요.
# data_cleaned.to_csv('clustering_dataset.csv', index=False, encoding='utf-8-sig')

# 5. 데이터 표준화 (스케일링)
# 데이터 범위가 다르므로 스케일링 필수 진행
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_cleaned)

# 6. Elbow Method 실행 및 수치 출력
inertia = []
k_range = range(1, 11)

print(f"{'K값':<5} | {'Inertia (오차 제곱합)':<20} | {'감소량':<15}")
print("-" * 45)

prev_inertia = None
for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    current_inertia = kmeans.inertia_
    
    drop = (prev_inertia - current_inertia) if prev_inertia is not None else 0
    print(f"{k:<6} | {current_inertia:<20.2f} | {drop:<15.2f}")
    
    inertia.append(current_inertia)
    prev_inertia = current_inertia

# 7. Elbow Method 시각화
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', color='darkblue', linewidth=2)
plt.title('Elbow Method for Optimal k (Adjusted Features)', fontsize=14)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()