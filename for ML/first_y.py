"""
STEP2 - 2트 시뮬레이션용
왜도 1.5 이상 log 적용 + 왜도 전후 출력
Y 생성 + h별 학습 데이터셋 생성
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────
# 0. 경로 설정
# ─────────────────────────────
BASE = r'C:\Users\82106\Desktop\2트_시뮬'
os.makedirs(os.path.join(BASE, '중간산출물'), exist_ok=True)

print("\n===== 2트 STEP2 (왜도 1.5 기준 log 적용) 시작 =====")

# ─────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────
df_derived = pd.read_csv(os.path.join(BASE,'파생변수_점수구성용.csv'), encoding='utf-8-sig')
df_density = pd.read_csv(os.path.join(BASE,'클러스터_밀도기초_파일.csv'), encoding='utf-8-sig')
df_cluster = pd.read_csv(os.path.join(BASE,'cluster_k3_행정동매핑.csv'), encoding='utf-8-sig')[['행정동_코드','cluster']]

print("\n데이터 로드 완료")
print("파생변수:", df_derived.shape)
print("밀도기초:", df_density.shape)
print("클러스터:", df_cluster.shape)

# ─────────────────────────────
# 2. Y 생성 (왜도 기반 log 선별 적용)
# ─────────────────────────────
print("\n[2] Y 생성 (왜도 ≥ 1.5 log 적용)")

CONSUME = ['유동1인당매출','등록1인당지출']
PRODUCE = ['유사_업종_점포_수','개업률']
POP = ['유동인구밀도','직장인구밀도','등록인구밀도']
ALL_Y = CONSUME + PRODUCE + POP

df_score = df_derived[['연도','분기','행정동_코드','행정동_코드_명'] + ALL_Y].copy()

# 1️⃣ log 적용 전 왜도 출력
print("\n[왜도 - log 적용 전]")
skew_before = df_score[ALL_Y].skew()
print(skew_before.sort_values(ascending=False))

# 2️⃣ log 대상 선정 (|왜도| ≥ 1.5, 비율 변수 제외)
LOG_TARGET = [
    col for col in ALL_Y
    if abs(skew_before[col]) >= 1.5 and col not in ['개업률']
]

print("\nlog 적용 대상 (|왜도| ≥ 1.5):")
print(LOG_TARGET)

# 3️⃣ log1p 적용
for col in LOG_TARGET:
    df_score[col] = np.log1p(df_score[col])

# 4️⃣ log 적용 후 왜도 출력
print("\n[왜도 - log 적용 후]")
skew_after = df_score[ALL_Y].skew()
print(skew_after.sort_values(ascending=False))

# 5️⃣ 분기 내 Z-score 계산
for col in ALL_Y:
    mean = df_score.groupby(['연도','분기'])[col].transform('mean')
    std  = df_score.groupby(['연도','분기'])[col].transform('std')
    df_score[f'Z_{col}'] = np.where(std>0,(df_score[col]-mean)/std,0)

df_score['소비점수'] = df_score[[f'Z_{c}' for c in CONSUME]].mean(axis=1)
df_score['생산점수'] = df_score[[f'Z_{c}' for c in PRODUCE]].mean(axis=1)
df_score['인구점수'] = df_score[[f'Z_{c}' for c in POP]].mean(axis=1)
df_score['경제활력점수'] = df_score[['소비점수','생산점수','인구점수']].mean(axis=1)

df_score['abs_q'] = (df_score['연도']-2020)*4 + (df_score['분기']-1)

y_path = os.path.join(BASE,'중간산출물','경제활력점수_전시점.csv')
df_score[['연도','분기','abs_q','행정동_코드','행정동_코드_명','경제활력점수']].to_csv(
    y_path,index=False,encoding='utf-8-sig'
)
print("\nY 저장 완료:", y_path)

# ─────────────────────────────
# 3. X 병합
# ─────────────────────────────
print("\n[3] X 병합")

X_DERIVED = [
    '유동1인당매출','등록1인당지출','유동인구밀도','직장인구밀도','등록인구밀도',
    '유동_등록비율','유동_직장비율','집객시설밀도','프랜차이즈비율',
    '임대시세_per_m2','유사_업종_점포_수','개업률'
]

X_DENSITY = ['폐업률','업종_밀도','총_집객시설_수']
ALL_X = X_DERIVED + X_DENSITY
KEY = ['연도','분기','행정동_코드','행정동_코드_명']

df_x = df_derived[KEY + X_DERIVED].merge(
    df_density[['연도','분기','행정동_코드'] + X_DENSITY],
    on=['연도','분기','행정동_코드'],
    how='left'
)

df_x['abs_q'] = (df_x['연도']-2020)*4 + (df_x['분기']-1)

df_x.replace([np.inf,-np.inf],np.nan,inplace=True)

# ─────────────────────────────
# 4. 구조적 0 처리
# ─────────────────────────────
print("\n[4] 구조적 0 처리")

ZERO_COLS = [
    '임대시세_per_m2',
    '직장인구밀도',
    '개업률',
    '폐업률',
    '유동_직장비율'
]

df_x = df_x.merge(df_cluster,on='행정동_코드',how='left')

for col in ZERO_COLS:
    if col in df_x.columns:
        df_x[col] = df_x[col].replace(0,np.nan)
        df_x[col] = df_x.groupby(['연도','분기','cluster'])[col].transform(lambda x: x.fillna(x.median()))
        df_x[col] = df_x.groupby(['연도','분기'])[col].transform(lambda x: x.fillna(x.median()))
        df_x[col].fillna(df_x[col].median(), inplace=True)

df_x.drop(columns=['cluster'], inplace=True)

print("\n남은 결측:")
print(df_x[ALL_X].isna().sum())

# ─────────────────────────────
# 5. h별 데이터 생성
# ─────────────────────────────
print("\n[5] h별 데이터 생성")

df_y_lookup = df_score[['행정동_코드','abs_q','경제활력점수']]

for h in [1,2,3,4]:
    print(f"\n--- h={h} ---")
    
    df_y_shift = df_y_lookup.copy()
    df_y_shift['abs_q'] -= h
    
    dataset = df_x.merge(
        df_y_shift.rename(columns={'경제활력점수':'Y'}),
        on=['행정동_코드','abs_q'],
        how='inner'
    )
    
    print("dropna 전:", len(dataset))
    dataset = dataset.dropna(subset=ALL_X+['Y'])
    print("dropna 후:", len(dataset))
    
    col_order = KEY + ['abs_q'] + ALL_X + ['Y']
    dataset = dataset[col_order].sort_values(['abs_q','행정동_코드']).reset_index(drop=True)
    
    save_path = os.path.join(BASE,'중간산출물',f'dataset_h{h}.csv')
    dataset.to_csv(save_path,index=False,encoding='utf-8-sig')
    print("저장 완료:", save_path)

print("\n===== 2트 STEP2 완료 =====")