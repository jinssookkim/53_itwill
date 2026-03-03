"""
STEP 2: Y 생성 및 h별 데이터셋 구성
=====================================
목적:
  1) 원본값 Z-score 기반으로 지역경제활력점수(Y) 생성
     - 차분(log_lag) 없이 각 (연도, 분기) 내에서 원본값을 바로 Z-score화
  2) 0값 처리: 클러스터 내 (연도, 분기) median으로 대체
  3) X(t 시점) + Y(t+h 시점) 매핑
  4) h=1,2,3,4 별 데이터셋 CSV 저장

입력 파일:
  - 파생변수_점수구성용.csv     : X변수 + Y 계산용 7개 지표 (9752행 x 18열)
  - 클러스터_밀도기초_파일.csv  : 추가 X변수 폐업률, 업종밀도 등 (9752행 x 41열)
  - cluster_k3_행정동매핑.csv  : 행정동별 클러스터 정보 (424행 x 7열)

출력 파일:
  - 중간산출물/경제활력점수_전시점.csv  : 전 시점 Y값
  - 중간산출물/dataset_h1.csv ~ h4.csv : h별 학습 데이터셋
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE = r'C:\Users\82106\Desktop\1트_시뮬'

os.makedirs(os.path.join(BASE, '중간산출물'), exist_ok=True)

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
print("[1] 데이터 로드 중...")

df_derived = pd.read_csv(
    os.path.join(BASE, '파생변수_점수구성용.csv'),
    encoding='utf-8-sig'
)
df_density = pd.read_csv(
    os.path.join(BASE, '클러스터_밀도기초_파일.csv'),
    encoding='utf-8-sig'
)
df_cluster = pd.read_csv(
    os.path.join(BASE, 'cluster_k3_행정동매핑.csv'),
    encoding='utf-8-sig'
)[['행정동_코드', 'cluster']]  # 클러스터 번호만 사용

print(f"  파생변수:  {df_derived.shape}")
print(f"  밀도기초:  {df_density.shape}")
print(f"  클러스터:  {df_cluster.shape}")
print(f"  클러스터별 동네 수:\n{df_cluster['cluster'].value_counts().sort_index().to_string()}")

# ─────────────────────────────────────────────
# 2. Y(지역경제활력점수) 생성
#
# 방식:
#   각 (연도, 분기) 그룹 내에서 7개 지표를 Z-score화
#   → 같은 분기 내 424개 행정동 간 상대적 위치
#   → 부문별(소비/생산/인구) 평균 → 최종 경제활력점수
#
# 기존 step1과의 차이:
#   기존: log(t+h) - log(t) → Z-score  (차분 후 표준화)
#   변경: 원본값 → Z-score             (차분 없이 현재 수준 기반)
# ─────────────────────────────────────────────
print("\n[2] Y(지역경제활력점수) 생성 중...")

# Y 계산에 사용할 7개 지표 정의
CONSUME_COLS    = ['유동1인당매출', '등록1인당지출']                # 소비 부문 (2개)
PRODUCE_COLS    = ['유사_업종_점포_수', '개업률']                   # 생산 부문 (2개)
POPULATION_COLS = ['유동인구밀도', '직장인구밀도', '등록인구밀도']  # 인구 부문 (3개)
ALL_Y_COLS      = CONSUME_COLS + PRODUCE_COLS + POPULATION_COLS    # 총 7개

df_score = df_derived[
    ['연도', '분기', '행정동_코드', '행정동_코드_명'] + ALL_Y_COLS
].copy()

# 각 (연도, 분기) 그룹 내에서 Z-score 계산
# → "이번 분기에 이 동네가 다른 동네들보다 얼마나 높냐"의 상대적 위치
for col in ALL_Y_COLS:
    grp_mean = df_score.groupby(['연도', '분기'])[col].transform('mean')
    grp_std  = df_score.groupby(['연도', '분기'])[col].transform('std')

    df_score[f'Z_{col}'] = np.where(
        grp_std > 0,
        (df_score[col] - grp_mean) / grp_std,
        0.0  # std=0이면 모든 동네가 같은 값 → Z=0으로 처리
    )

# 부문별 Z-score 평균 계산
df_score['소비점수']    = df_score[[f'Z_{c}' for c in CONSUME_COLS]].mean(axis=1)
df_score['생산점수']    = df_score[[f'Z_{c}' for c in PRODUCE_COLS]].mean(axis=1)
df_score['인구점수']    = df_score[[f'Z_{c}' for c in POPULATION_COLS]].mean(axis=1)

# 최종 경제활력점수 = 3개 부문 단순 평균
df_score['경제활력점수'] = (
    df_score['소비점수'] + df_score['생산점수'] + df_score['인구점수']
) / 3

# abs_q: 절대 분기 번호 (2020Q1=0, 2020Q2=1, ..., 2025Q3=22)
df_score['abs_q'] = (df_score['연도'] - 2020) * 4 + (df_score['분기'] - 1)

print(f"  경제활력점수 기술통계:")
print(f"    mean = {df_score['경제활력점수'].mean():.4f}")
print(f"    std  = {df_score['경제활력점수'].std():.4f}")
print(f"    min  = {df_score['경제활력점수'].min():.4f}")
print(f"    max  = {df_score['경제활력점수'].max():.4f}")

# Y값 전체 저장 (나중에 클러스터별 비교 등에 활용)
y_save_cols = ['연도', '분기', 'abs_q', '행정동_코드', '행정동_코드_명', '경제활력점수']
y_path = os.path.join(BASE, '중간산출물', '경제활력점수_전시점.csv')
df_score[y_save_cols].to_csv(y_path, index=False, encoding='utf-8-sig')
print(f"\n  → Y 저장 완료: {y_path}")

# ─────────────────────────────────────────────
# 3. X변수 구성 및 병합
# ─────────────────────────────────────────────
print("\n[3] X변수 구성 중...")

# 파생변수에서 가져올 X 컬럼 (12개)
X_DERIVED_COLS = [
    '유동1인당매출',     # 소비: 유동인구 1인당 추정매출
    '등록1인당지출',     # 소비: 등록인구 1인당 지출액
    '유동인구밀도',      # 인구: 면적당 유동인구 수
    '직장인구밀도',      # 인구: 면적당 직장인구 수 (0값 존재 → 처리 필요)
    '등록인구밀도',      # 인구: 면적당 등록인구 수
    '유동_등록비율',     # 비율: 유동인구 / 등록인구
    '유동_직장비율',     # 비율: 유동인구 / 직장인구
    '집객시설밀도',      # 상권: 면적당 집객시설 수
    '프랜차이즈비율',    # 상권: 전체 점포 중 프랜차이즈 비율
    '임대시세_per_m2',   # 상권: m2당 임대시세 (0값 존재 → 처리 필요)
    '유사_업종_점포_수', # 생산: 유사 업종 경쟁 점포 수
    '개업률',            # 생산: 신규 개업 비율 (0값 존재 → 처리 필요)
]

# 밀도기초에서 가져올 X 컬럼 (3개, Y에 포함되지 않은 변수)
X_DENSITY_COLS = [
    '폐업률',          # 생산: 폐업 비율 (0값 존재 → 처리 필요)
    '업종_밀도',       # 밀도: 면적당 업종 수
    '총_집객시설_수',  # 밀도: 전체 집객시설 수
]

ALL_X_COLS = X_DERIVED_COLS + X_DENSITY_COLS  # 총 15개

KEY_COLS = ['연도', '분기', '행정동_코드', '행정동_코드_명']

# 파생변수 + 밀도기초 병합 (연도, 분기, 행정동_코드 기준)
df_x = df_derived[KEY_COLS + X_DERIVED_COLS].merge(
    df_density[['연도', '분기', '행정동_코드'] + X_DENSITY_COLS],
    on=['연도', '분기', '행정동_코드'],
    how='left'
)

# abs_q 추가
df_x['abs_q'] = (df_x['연도'] - 2020) * 4 + (df_x['분기'] - 1)

print(f"  X 데이터 shape: {df_x.shape}")
print(f"  X 변수 {len(ALL_X_COLS)}개")

# ─────────────────────────────────────────────
# 4. 0값 처리: 클러스터 내 (연도, 분기) median으로 대체
#
# 0값이 의미하는 것:
#   임대시세_per_m2 = 0 → 임대 데이터 없음 (진짜 0원이 아님)
#   직장인구밀도    = 0 → 직장인구 데이터 없음
#   개업률, 폐업률  = 0 → 점포 수 부족으로 데이터 없음
#
# 대체 방식:
#   1순위: 같은 (연도, 분기, cluster) 내 median
#          → 비슷한 성격의 동네들 기준으로 채움
#          예) C2 주거밀착형 동네의 결측 → C2 동네들의 median으로 대체
#   2순위: 클러스터 내 전부 결측인 경우
#          → 같은 (연도, 분기) 전체 median으로 보완
# ─────────────────────────────────────────────
print("\n[4] 0값 처리 중 (클러스터 내 median 대체)...")

# 클러스터 정보 임시 병합 (대체 처리 후 제거할 것)
df_x = df_x.merge(df_cluster, on='행정동_코드', how='left')

# 0값 처리 대상 컬럼
ZERO_TO_NAN_COLS = ['임대시세_per_m2', '직장인구밀도', '개업률', '폐업률']

for col in ZERO_TO_NAN_COLS:
    if col not in df_x.columns:
        continue

    zero_count = (df_x[col] == 0).sum()
    if zero_count == 0:
        print(f"  {col}: 0값 없음 → 처리 불필요")
        continue

    print(f"  {col}: 0값 {zero_count}개 처리 중...")

    # Step 1: 0 → NaN으로 변환 (모델이 진짜 0으로 인식하지 않도록)
    df_x[col] = df_x[col].replace(0, np.nan)

    # Step 2: 같은 (연도, 분기, cluster) 그룹 내 median으로 대체
    #         → 같은 시점 + 같은 유형 동네들의 중앙값으로 채움
    df_x[col] = df_x.groupby(
        ['연도', '분기', 'cluster']
    )[col].transform(lambda x: x.fillna(x.median()))

    # Step 3: 혹시 클러스터 내 전부 NaN인 경우
    #         → 같은 (연도, 분기) 전체 median으로 보완
    remaining = df_x[col].isna().sum()
    if remaining > 0:
        df_x[col] = df_x.groupby(
            ['연도', '분기']
        )[col].transform(lambda x: x.fillna(x.median()))
        print(f"    ※ {remaining}개는 클러스터 내 전부 결측 → 전체 분기 median으로 보완")

    print(f"    → 완료 (남은 결측: {df_x[col].isna().sum()}개)")

# 클러스터 컬럼 제거 (X변수로 사용 안 하기로 결정)
df_x = df_x.drop(columns=['cluster'])

# ─────────────────────────────────────────────
# 5. h별 데이터셋 생성
#
# 핵심 로직:
#   X(t 시점 피처) + Y(t+h 시점 경제활력점수) 매핑
#
#   h=1 예시:
#     X(2020Q1, 행정동A) → Y(2020Q2, 행정동A)
#     X(2020Q2, 행정동A) → Y(2020Q3, 행정동A)
#     ...
#     X(2025Q2, 행정동A) → Y(2025Q3, 행정동A)  ← 학습 마지막 행
#
#   h=4 예시:
#     X(2020Q1, 행정동A) → Y(2021Q1, 행정동A)
#     ...
#     X(2024Q3, 행정동A) → Y(2025Q3, 행정동A)  ← 학습 마지막 행
#
#   예측 시 (학습 후):
#     X(2025Q3) 입력 → h=1: 2025Q4 예측
#                    → h=2: 2026Q1 예측
#                    → h=3: 2026Q2 예측
#                    → h=4: 2026Q3 예측
# ─────────────────────────────────────────────
print("\n[5] h별 데이터셋 생성 중...")

# Y 조회용 테이블: 행정동_코드 + abs_q → 경제활력점수
df_y_lookup = df_score[['행정동_코드', 'abs_q', '경제활력점수']].copy()

for h in [1, 2, 3, 4]:
    print(f"\n  ▶ h={h} ({h*3}개월 후 예측)")

    # Y의 abs_q를 h만큼 당겨서 X의 abs_q와 맞춤
    # 예) h=1:
    #   원래 Y(abs_q=1) → abs_q를 0으로 바꿈 → X(abs_q=0)과 병합
    #   원래 Y(abs_q=2) → abs_q를 1로 바꿈  → X(abs_q=1)과 병합
    df_y_shifted = df_y_lookup.copy()
    df_y_shifted['abs_q'] = df_y_shifted['abs_q'] - h

    # X(t)와 Y(t+h) 병합 (행정동_코드 + abs_q 동시 일치하는 것만)
    dataset = df_x.merge(
        df_y_shifted[['행정동_코드', 'abs_q', '경제활력점수']].rename(
            columns={'경제활력점수': 'Y'}
        ),
        on=['행정동_코드', 'abs_q'],
        how='inner'
    )

    # 결측 제거
    before = len(dataset)
    dataset = dataset.dropna(subset=ALL_X_COLS + ['Y'])
    after = len(dataset)

    print(f"    행 수: {before:,}행 → 결측 제거 후 {after:,}행")
    print(f"    X 시점 범위: "
          f"{dataset['연도'].min()}Q{dataset['분기'].min()} "
          f"~ {dataset['연도'].max()}Q{dataset['분기'].max()}")
    print(f"    Y 기술통계: "
          f"mean={dataset['Y'].mean():.4f}, "
          f"std={dataset['Y'].std():.4f}, "
          f"min={dataset['Y'].min():.4f}, "
          f"max={dataset['Y'].max():.4f}")

    # 컬럼 순서 정리: 메타정보 + X변수 + Y
    col_order = KEY_COLS + ['abs_q'] + ALL_X_COLS + ['Y']
    dataset = dataset[col_order].sort_values(
        ['abs_q', '행정동_코드']
    ).reset_index(drop=True)

    # 저장
    save_path = os.path.join(BASE, '중간산출물', f'dataset_h{h}.csv')
    dataset.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"    → 저장 완료: {save_path}")

# ─────────────────────────────────────────────
# 6. 최종 요약
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("h별 데이터셋 요약")
print("=" * 60)
print(f"  {'h':>2} | {'예상 행수':>10} | {'X 마지막 시점':>14} | {'Y 마지막 시점':>14} | {'예측 목표':>10}")
print("  " + "-" * 58)

summary = {
    1: (424*22, '2025Q2', '2025Q3', '2025Q4'),
    2: (424*21, '2025Q1', '2025Q3', '2026Q1'),
    3: (424*20, '2024Q4', '2025Q3', '2026Q2'),
    4: (424*19, '2024Q3', '2025Q3', '2026Q3'),
}
for h, (rows, x_last, y_last, pred_target) in summary.items():
    print(f"  {h:>2} | {rows:>10,} | {x_last:>14} | {y_last:>14} | {pred_target:>10}")

print("\n※ 다음 단계(step3)에서:")
print("   dataset_h1~h4.csv 로 모델 학습")
print("   → X(2025Q3) 입력 → h별 미래 경제활력점수 예측")

print("\n" + "=" * 60)
print("STEP 2 완료: 경제활력점수_전시점.csv + dataset_h1~h4.csv 저장됨")
print("=" * 60)