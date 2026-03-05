"""
7트_시뮬_1 STEP1 - X변수 19개 버전 (22개 → 19개)
- 제외: 점당매출기여도, 인구당소비집중도, IS_WORKER_MISSING
- 유지: 임대료대비수익성, IS_RENT_MISSING
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

BASE = r'C:\Users\82106\Desktop\7트_시뮬_1'
OUT  = os.path.join(BASE, '중간산출물')
os.makedirs(OUT, exist_ok=True)

print("===== 7트_시뮬_1 STEP1 (X=19개) 시작 =====")

# ─────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────
df1 = pd.read_csv(os.path.join(BASE, '파생변수_점수구성용.csv'), encoding='utf-8-sig')
df2 = pd.read_csv(os.path.join(BASE, '클러스터_밀도기초_파일.csv'), encoding='utf-8-sig')
df_cluster = pd.read_csv(os.path.join(BASE, 'cluster_k3_행정동매핑.csv'), encoding='utf-8-sig')[['행정동_코드','cluster']]

DENSITY_COLS = [
    '개업_점포_수','폐업_점포_수','프랜차이즈_점포_수','폐업률',
    '총_집객시설_수','관공서_수','은행_수','종합병원_수','일반_병원_수','약국_수',
    '유치원_수','초등학교_수','중학교_수','고등학교_수','대학교_수',
    '백화점_수','슈퍼마켓_수','극장_수','숙박_시설_수',
    '철도_역_수','지하철_역_수','버스_정거장_수','버스_터미널_수','공항_수',
    '총_유동인구_수','총_등록인구_수','총_직장_인구_수',
    '추정매출액','전체_임대시세','지출총금액',
    'IS_WORKER_MISSING','IS_RENT_MISSING',
]

available  = [c for c in DENSITY_COLS if c in df2.columns]
already    = set(df1.columns) - {'연도','분기','행정동_코드','행정동_코드_명'}
merge_cols = [c for c in available if c not in already]

df = df1.merge(
    df2[['연도','분기','행정동_코드'] + merge_cols],
    on=['연도','분기','행정동_코드'], how='left'
)
df['abs_q'] = (df['연도'] - 2020) * 4 + (df['분기'] - 1)
df = df.merge(df_cluster, on='행정동_코드', how='left')
df.replace([np.inf, -np.inf], np.nan, inplace=True)

print(f"데이터 shape: {df.shape}")
assert len(df) == len(df1), "❌ 병합 후 행수 불일치!"
print("✅ 병합 행수 일치")

# ─────────────────────────────
# 2. 결측치 처리 (diff 계산 전)
# ─────────────────────────────
print("\n[2] 결측치 처리")
FILL_COLS = ['직장인구밀도','임대시세_per_m2','유동_등록비율','유동_직장비율','전체_임대시세']
for col in FILL_COLS:
    if col not in df.columns:
        continue
    df[col] = df[col].replace(0, np.nan)
    df[col] = df.groupby(['연도','분기','cluster'])[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df.groupby(['연도','분기'])[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())
    print(f"  {col}: 처리 완료")

df = df.drop(columns=['cluster'])

# ─────────────────────────────
# 3. Y 생성 (기존과 동일)
# ─────────────────────────────
print("\n[3] Y 생성")
ALL_Y = ['유동1인당매출','등록1인당지출','유사_업종_점포_수','개업률',
         '유동인구밀도','직장인구밀도','등록인구밀도']

for col in ALL_Y:
    df[col+'_log'] = np.log1p(df[col])
    mean = df.groupby(['연도','분기'])[col+'_log'].transform('mean')
    std  = df.groupby(['연도','분기'])[col+'_log'].transform('std')
    df['Z_'+col] = np.where(std > 0, (df[col+'_log'] - mean) / std, 0)

df['소비점수']     = df[['Z_유동1인당매출','Z_등록1인당지출']].mean(axis=1)
df['생산점수']     = df[['Z_유사_업종_점포_수','Z_개업률']].mean(axis=1)
df['인구점수']     = df[['Z_유동인구밀도','Z_직장인구밀도','Z_등록인구밀도']].mean(axis=1)
df['경제활력점수'] = df[['소비점수','생산점수','인구점수']].mean(axis=1)
print(f"  평균: {df['경제활력점수'].mean():.4f} | 표준편차: {df['경제활력점수'].std():.4f}")

# ─────────────────────────────
# 4. 집객시설 그룹핑
# ─────────────────────────────
print("\n[4] 집객시설 그룹핑")
df['교통시설_수'] = df[['지하철_역_수','버스_정거장_수','철도_역_수','버스_터미널_수','공항_수']].sum(axis=1)
df['의료시설_수'] = df[['종합병원_수','일반_병원_수','약국_수']].sum(axis=1)
df['교육시설_수'] = df[['유치원_수','초등학교_수','중학교_수','고등학교_수','대학교_수']].sum(axis=1)
df['상업시설_수'] = df[['백화점_수','슈퍼마켓_수','극장_수','숙박_시설_수']].sum(axis=1)
df['공공시설_수'] = df[['관공서_수','은행_수']].sum(axis=1)
print("  완료")

# ─────────────────────────────
# 5. 효율성 지표 (임대료대비수익성만 유지)
# ─────────────────────────────
print("\n[5] 효율성 지표 생성 (임대료대비수익성만)")
df['임대료대비수익성'] = df['추정매출액'] / df['전체_임대시세'].replace(0, np.nan)
print("  완료")

# ─────────────────────────────
# 6. 모멘텀 diff 변수 생성
# ─────────────────────────────
print("\n[6] 모멘텀 diff 변수 생성")
df = df.sort_values(['행정동_코드','연도','분기']).reset_index(drop=True)

for col in ['개업률','추정매출액','총_유동인구_수']:
    if col not in df.columns:
        print(f"  {col} 없음, 스킵")
        continue
    df[f'{col}_QoQ_diff']   = df.groupby('행정동_코드')[col].diff(1)
    df[f'{col}_YoY_diff']   = df.groupby('행정동_코드')[col].diff(4)
    df[f'{col}_diff_roll4'] = df.groupby('행정동_코드')[f'{col}_QoQ_diff'].transform(
        lambda x: x.rolling(4, min_periods=2).mean()
    )
    print(f"  {col}: QoQ_diff / YoY_diff / diff_roll4 생성")

# ─────────────────────────────
# 7. X변수 구성 (✅ 19개 고정)
# ─────────────────────────────
print("\n[7] X변수 구성 (19개)")
X_VARS = [
    # 구조적 기초
    '프랜차이즈_점포_수', '임대시세_per_m2', '개업_점포_수',
    # 집객시설 그룹핑
    '교통시설_수', '의료시설_수', '교육시설_수', '상업시설_수', '공공시설_수',
    # 효율성 지표 (Y 재료 포함 버전)
    '임대료대비수익성',
    # 모멘텀 diff
    '개업률_QoQ_diff',         '개업률_YoY_diff',         '개업률_diff_roll4',
    '추정매출액_QoQ_diff',     '추정매출액_YoY_diff',     '추정매출액_diff_roll4',
    '총_유동인구_수_QoQ_diff', '총_유동인구_수_YoY_diff', '총_유동인구_수_diff_roll4',
    # 더미변수
    'IS_RENT_MISSING',
]
X_VARS = [c for c in X_VARS if c in df.columns]

print(f"  X변수 총 {len(X_VARS)}개:")
for v in X_VARS:
    print(f"    {v}")

# ─────────────────────────────
# 8. h별 데이터셋 생성 + Y 매핑 검증
# ─────────────────────────────
print("\n[8] h별 데이터셋 생성")
KEY = ['연도','분기','행정동_코드','행정동_코드_명']
df_y_lookup = df[['행정동_코드','abs_q','경제활력점수']].copy()

for h in [1, 2, 3, 4]:
    df_y_shift = df_y_lookup.copy()
    df_y_shift['abs_q'] -= h  # X abs_q에 Y(t+h)를 붙이기 위해 Y abs_q를 당김

    dataset = df[KEY + ['abs_q'] + X_VARS].merge(
        df_y_shift.rename(columns={'경제활력점수':'Y'}),
        on=['행정동_코드','abs_q'], how='inner'
    )
    dataset = dataset.dropna(subset=X_VARS + ['Y'])
    dataset = dataset.sort_values(['abs_q','행정동_코드']).reset_index(drop=True)

    # Y 매핑 검증
    last_x     = dataset.iloc[-1]
    x_abs      = int(last_x['abs_q'])
    y_abs      = x_abs + h
    y_year     = 2020 + y_abs // 4
    y_q        = y_abs % 4 + 1
    x_year_chk = 2020 + x_abs // 4
    x_q_chk    = x_abs % 4 + 1

    path = os.path.join(OUT, f'dataset_h{h}.csv')
    dataset.to_csv(path, index=False, encoding='utf-8-sig')

    q_min = int(dataset['abs_q'].min())
    q_max = int(dataset['abs_q'].max())
    print(f"  h={h}: {len(dataset)}행 | "
          f"X시점 {2020+q_min//4}Q{q_min%4+1}~{2020+q_max//4}Q{q_max%4+1} | "
          f"X변수 {len(X_VARS)}개 | "
          f"Y매핑검증: X={x_year_chk}Q{x_q_chk} → Y={y_year}Q{y_q} ✅")

print("\n===== 7트_시뮬_1 STEP1 (X=19개) 완료 =====")
print("다음: STEP2 실행 (step2_model_fi.py)")