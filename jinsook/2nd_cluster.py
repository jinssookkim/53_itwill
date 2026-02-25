

import pandas as pd
import numpy as np


path = 'https://github.com/jinssookkim/53_itwill/raw/refs/heads/main/%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0/final_%EC%BB%AC%EB%9F%BC%EC%B6%94%EA%B0%80%EC%99%84%EB%A3%8C.csv'
df = pd.read_csv(path)

df.info()


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



# ── 한글 폰트 설정 ──────────────────────────────────────────
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False



# ── 2. 2022년 이후 데이터 필터링 후 행정동별 집계 ──────────
df_recent1 = df[df['연도'] >= 2022]
agg_df = df_recent1.groupby('행정동_코드_명').mean(numeric_only=True).reset_index()


df_recent1.info()
agg_df.info()


# 세대별 매출 비율
agg_df['MZ_매출_비율'] = (agg_df['연령대_10_매출_금액'] + agg_df['연령대_20_매출_금액'] + agg_df['연령대_30_매출_금액']) / agg_df['추정매출액']

agg_df['중년_매출_비율'] = (agg_df['연령대_40_매출_금액'] + agg_df['연령대_50_매출_금액']) / agg_df['추정매출액']

agg_df['노년_매출_비율'] = agg_df['연령대_60_이상_매출_금액'] / agg_df['추정매출액']


# 주중/주말 비율
agg_df['주중_비율'] = agg_df['주중_매출_금액'] / agg_df['추정매출액']

agg_df[['주중_비율', '주중_매출_금액', '추정매출액']]

agg_df['주말_비율'] = agg_df['주말_매출_금액'] / agg_df['추정매출액']


# 주말/연령 매출 비율
agg_df['MZ_주말_매출_비율'] = (agg_df['연령대_10_매출_금액'] + agg_df['연령대_20_매출_금액'] + agg_df['연령대_30_매출_금액']) / agg_df['주말_매출_금액']
agg_df['중년_주말_매출_비율'] = (agg_df['연령대_40_매출_금액'] + agg_df['연령대_50_매출_금액']) / agg_df['주말_매출_금액']

# 직장 매출 비율
agg_df['직장_주중_매출_비율'] = (agg_df['연령대_30_매출_금액'] + agg_df['연령대_40_매출_금액'] + agg_df['연령대_50_매출_금액']) / agg_df['주중_매출_금액']

agg_df.info()

# ── 3. 군집에 사용할 소비 변수 정의 ─────────────────────────────
# 세대별 매출 특성
age_sales_cols = [
    'MZ_매출_비율', '중년_매출_비율', '노년_매출_비율'
]

# 요일별 매출 특성
week_cols = [
    '주중_비율', '주말_비율'
]

# 주말 매출 특성
weekend_cols = [
    'MZ_주말_매출_비율', '중년_주말_매출_비율'
]

# 직장인 매출 특성
worker_cols = [
    '직장_주중_매출_비율'
]

feature_cols = age_sales_cols + week_cols + weekend_cols + worker_cols

# 실제 존재하는 컬럼만 필터링
feature_cols = [c for c in feature_cols if c in agg_df.columns]
print(f"사용 변수 수: {len(feature_cols)}개")

# ── 4. 결측치 처리 & 스케일링 ──────────────────────────────
X = agg_df[feature_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 5. 최적 군집 수 탐색 (Elbow + Silhouette) ──────────────
k_range = range(4, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

best_k = k_range.start + silhouette_scores.index(max(silhouette_scores))
print(f"\n✅ 최적 군집 수 (Silhouette 기준): {best_k}")

# ── 6. 시각화: Elbow + Silhouette ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(k_range, inertias, 'bo-')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('군집 수 (k)')
axes[0].set_ylabel('Inertia')
axes[0].axvline(x=best_k, color='red', linestyle='--', label=f'best k={best_k}')
axes[0].legend()

axes[1].plot(k_range, silhouette_scores, 'go-')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('군집 수 (k)')
axes[1].set_ylabel('Score')
axes[1].axvline(x=best_k, color='red', linestyle='--', label=f'best k={best_k}')
axes[1].legend()

plt.tight_layout()
plt.savefig('optimal_k.png', dpi=150)
plt.show()

# ── 7. 최적 k로 최종 군집화 ────────────────────────────────
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
agg_df['cluster'] = kmeans_final.fit_predict(X_scaled)

# ── 8. PCA 2D 시각화 ───────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=agg_df['cluster'], cmap='tab10', alpha=0.7, s=60)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({explained[0]:.1f}%)')
plt.ylabel(f'PC2 ({explained[1]:.1f}%)')
plt.title(f'행정동 군집 분포 (PCA 2D) - k={best_k}')

for i, row in agg_df.iterrows():
    plt.annotate(row['행정동_코드_명'], (X_pca[i, 0], X_pca[i, 1]),
                 fontsize=6, alpha=0.6)

plt.tight_layout()
plt.savefig('cluster_pca.png', dpi=150)
plt.show()

# ── 9. 군집별 특성 요약 ────────────────────────────────────
cluster_summary = agg_df.groupby('cluster')[feature_cols].mean()
print("\n📊 군집별 평균값:")
print(cluster_summary.T.to_string())


# 클러스터 안에 있는 행정동 확인

for k in sorted(agg_df['cluster'].unique()):
    dongs = agg_df[agg_df['cluster'] == k]['행정동_코드_명'].tolist()
    print(f"\n📍 Cluster {k} ({len(dongs)}개 행정동)")
    print(dongs)


    
# ── 10. 결과 저장 ──────────────────────────────────────────
result = agg_df[['행정동_코드_명', 'cluster'] + feature_cols]
result.to_csv('cluster_result.csv', index=False, encoding='utf-8-sig')
print("\n✅ 결과 저장 완료: cluster_result.csv")