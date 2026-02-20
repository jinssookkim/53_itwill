import pandas as pd
import numpy as np

# ===============================
# 1) 데이터 로드 + 컬럼 정리
# ===============================
df = pd.read_csv("final_all_merge.csv")

# 컬럼명 공백/숨은 공백 제거 (필수)
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "", regex=False)
)

# ===============================
# 2) 변수 정의 (네 실제 컬럼 기준)
# ===============================
consumption_vars = ["추정매출액", "지출총금액"]
production_vars  = ["유사_업종_점포_수", "개업_점포_수", "개업률"]
population_vars  = ["총_유동인구_수", "총_상주인구_수", "총_직장인구_수"]

all_vars = consumption_vars + production_vars + population_vars

# 컬럼 존재 체크
missing = [c for c in ["연도", "분기", "행정동_코드", "행정동_코드_명"] + all_vars if c not in df.columns]
if missing:
    raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

# 숫자형 변환 (문자열이면 계산 터짐 방지)
for c in all_vars:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ===============================
# 3) 분기별 z-score 표준화 (apply 대신 transform)
#    z = (x - mean) / std
# ===============================
g = df.groupby(["연도", "분기"])

means = g[all_vars].transform("mean")
stds  = g[all_vars].transform("std").replace(0, np.nan)  # 분산 0이면 NaN 처리

z_df = (df[all_vars] - means) / stds

# 표준화된 값 컬럼 덮어쓰기 (원하면 _z 붙여도 됨)
df[all_vars] = z_df

# ===============================
# 4) 부문 점수 + 종합 점수
# ===============================
df["소비점수"] = df[consumption_vars].mean(axis=1)
df["생산점수"] = df[production_vars].mean(axis=1)
df["인구점수"] = df[population_vars].mean(axis=1)

df["지역경제활력점수"] = df[["소비점수", "생산점수", "인구점수"]].mean(axis=1)

# (선택) 국토부 스타일 T점수
df["지역경제활력점수_T"] = 50 + 10 * df["지역경제활력점수"]

# ===============================
# 5) 결과 저장
# ===============================
result_cols = [
    "연도", "분기", "구이름",
    "행정동_코드", "행정동_코드_명",
    "소비점수", "생산점수", "인구점수",
    "지역경제활력점수", "지역경제활력점수_T"
]

# 구이름 없을 수도 있으니 안전 처리
result_cols = [c for c in result_cols if c in df.columns]

out_path = "행정동_연분기별_지역경제활력점수.csv"
df[result_cols].to_csv(out_path, index=False, encoding="utf-8-sig")

print("✅ 완료:", out_path)
print("컬럼 확인:", result_cols)