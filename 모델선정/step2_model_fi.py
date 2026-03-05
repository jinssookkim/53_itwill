"""
7트_시뮬_1 STEP2 v3 - Time Series Cross-Validation + 최종 Test 평가
══════════════════════════════════════════════════════════════════════

[분할 전략]

  ┌─ CV (모델 선택 & PI 계산) ──────────────────────────────────┐
  │  TimeSeriesSplit(n_splits=4)                                 │
  │                                                              │
  │  Fold1: Train[2020Q1~2022Q1] → Val[2022Q2~2022Q4]          │
  │  Fold2: Train[2020Q1~2022Q4] → Val[2023Q1~2023Q2]          │
  │  Fold3: Train[2020Q1~2023Q2] → Val[2023Q3~2024Q1]          │
  │  Fold4: Train[2020Q1~2024Q1] → Val[2024Q2~2024Q3]          │
  │                                                              │
  │  → 4번 평균 성능으로 모델 선택 & 변수 중요도 판단           │
  └─────────────────────────────────────────────────────────────┘

  ┌─ 최종 Test (딱 한 번만) ───────────────────────────────────┐
  │  Train: 전체 데이터 중 마지막 N개 분기 제외한 전부          │
  │  Test : 마지막 N개 분기 (2024Q4~2025Q3)                    │
  │  → CV에서 선택된 최고 모델을 여기서 최종 평가              │
  └─────────────────────────────────────────────────────────────┘

[왜 이렇게 하는가]
  - 일반 Train/Test 2분할:
      모델 선택할 때 Test를 반복해서 봄 → Test 오염 → 수치 못 믿음
  - Train/Val/Test 3분할:
      Val이 4분기밖에 안 됨 → 어떤 분기가 걸리냐에 따라 결과 요동
  - 일반 K-Fold:
      미래 데이터로 과거 예측하는 상황 발생 → 시계열에 절대 불가
  - ✅ Time Series Split + 최종 Test:
      항상 과거→미래 방향 유지, 4번 평균으로 운빨 제거,
      Test는 딱 한 번만 써서 오염 없음

평가지표: Spearman, Top-K Hit Rate, Direction Accuracy, RMSE, R²
PI      : CV Fold별 PI 평균 → 앙상블 PI
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

from sklearn.metrics       import r2_score
from sklearn.inspection    import permutation_importance
from sklearn.ensemble      import HistGradientBoostingRegressor
from sklearn.linear_model  import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats           import spearmanr, skew
import lightgbm as lgb
from xgboost   import XGBRegressor
from catboost  import CatBoostRegressor

# ── 한글 폰트 ──────────────────────────────────────────────────────
_available = {f.name for f in fm.fontManager.ttflist}
for _f in ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'DejaVu Sans']:
    if _f in _available:
        plt.rcParams['font.family'] = _f
        break
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════════
# 경로 설정
# ══════════════════════════════════════════════════════════════════
BASE = r'C:\Users\82106\Desktop\7트_시뮬_1'
MID  = os.path.join(BASE, '중간산출물')
OUT  = os.path.join(MID,  'model_results_v3')
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 설정값
# ══════════════════════════════════════════════════════════════════
N_SPLITS     = 4       # TimeSeriesSplit fold 수
TEST_Q       = 4       # 마지막 몇 개 분기를 Test로 보존 (약 2024Q4~2025Q3)
PI_THRESHOLD = 0.001
TOP_K        = 20

# ══════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════
def abs_to_label(a):
    return f"{2020 + int(a)//4}Q{int(a)%4+1}"

def evaluate_all(y_true, y_pred, name, split_label):
    r2   = r2_score(y_true, y_pred)
    sp   = spearmanr(y_true, y_pred).correlation
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    k         = min(TOP_K, len(y_true))
    true_topk = set(np.argsort(y_true)[-k:])
    pred_topk = set(np.argsort(y_pred)[-k:])
    topk_hit  = len(true_topk & pred_topk) / k

    dir_true = (y_true > np.median(y_true)).astype(int)
    dir_pred = (y_pred > np.median(y_pred)).astype(int)
    dir_acc  = np.mean(dir_true == dir_pred)

    return {
        'split':     split_label,
        'model':     name,
        'Spearman':  round(float(sp),       4),
        'Top_K_Hit': round(float(topk_hit), 4),
        'Dir_Acc':   round(float(dir_acc),  4),
        'RMSE':      round(float(rmse),     4),
        'R2':        round(float(r2),       4),
    }

# ══════════════════════════════════════════════════════════════════
# 모델 정의
# ══════════════════════════════════════════════════════════════════
def get_models():
    return {
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1),
        'XGBoost': XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            random_state=42, verbosity=0),
        'CatBoost': CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6,
            random_seed=42, verbose=0),
        'HistGBM': HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.05, random_state=42),
        'Ridge': Ridge(alpha=1.0),
    }

# ══════════════════════════════════════════════════════════════════
# h별 실행
# ══════════════════════════════════════════════════════════════════
all_cv_summary   = []   # CV fold별 성능 (모델 선택용)
all_test_summary = []   # 최종 Test 성능 (보고용)

for h in [1, 2, 3, 4]:
    print(f"\n{'═'*70}")
    print(f"  h={h}  ({['2025Q4','2026Q1','2026Q2','2026Q3'][h-1]} 예측)")
    print(f"{'═'*70}")

    df = pd.read_csv(
        os.path.join(MID, f'dataset_h{h}.csv'), encoding='utf-8-sig'
    ).sort_values('abs_q').reset_index(drop=True)

    X_COLS = [c for c in df.columns
              if c not in ['Y','연도','분기','행정동_코드','행정동_코드_명','abs_q']]
    print(f"  X변수: {len(X_COLS)}개 | 전체: {len(df)}행")

    # ── log 변환 (Train 왜도 기준으로 결정, 전체 적용) ───────────────
    # 주의: Test 데이터를 보기 전에 변환 여부를 결정해야 함
    #       → CV 전체 제외한 Test 제외 데이터로 왜도 판단
    SKIP_LOG    = [c for c in X_COLS if any(k in c for k in ['QoQ','YoY','diff'])]
    LOG_TARGETS = [c for c in X_COLS if c not in SKIP_LOG]

    unique_q   = sorted(df['abs_q'].unique())
    n_q        = len(unique_q)
    test_q_cut = unique_q[-TEST_Q]   # 마지막 TEST_Q개 분기가 Test

    df_cv   = df[df['abs_q'] <  test_q_cut].copy()   # CV에 쓸 데이터
    df_test = df[df['abs_q'] >= test_q_cut].copy()   # 절대 CV에 안 쓰는 Test

    log_applied = []
    for col in LOG_TARGETS:
        if col in df_cv.columns and skew(df_cv[col].dropna()) >= 1.5:
            df_cv[col]   = np.log1p(df_cv[col])
            df_test[col] = np.log1p(df_test[col])
            log_applied.append(col)
    print(f"  log 변환: {len(log_applied)}개")

    # CV / Test 시점 출력
    cv_q_vals   = sorted(df_cv['abs_q'].unique())
    test_q_vals = sorted(df_test['abs_q'].unique())
    print(f"  CV 범위  : {abs_to_label(cv_q_vals[0])} ~ {abs_to_label(cv_q_vals[-1])}  ({len(df_cv)}행)")
    if len(test_q_vals):
        print(f"  Test 범위: {abs_to_label(test_q_vals[0])} ~ {abs_to_label(test_q_vals[-1])}  ({len(df_test)}행)")
    else:
        print(f"  ⚠️  Test 데이터 없음 (h={h})")

    # ── TimeSeriesSplit: 분기(abs_q) 단위로 split ────────────────────
    # 행 단위가 아닌 '분기' 단위로 fold를 나눔
    # → 같은 분기 내 행정동들이 Train/Val에 섞이지 않도록
    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    cv_q_arr = np.array(cv_q_vals)  # 분기 배열

    print(f"\n  [Time Series CV — {N_SPLITS} Folds]")
    print(f"  {'Fold':<6} {'Train 범위':<22} {'Val 범위':<22} {'행수(tr/val)'}")
    print(f"  {'-'*65}")

    # fold별 성능 누적
    fold_metrics  = {name: [] for name in get_models()}
    fold_pi       = {name: [] for name in get_models()}   # fold별 PI 누적

    for fold_idx, (tr_q_idx, val_q_idx) in enumerate(tss.split(cv_q_arr), 1):
        tr_q_set  = set(cv_q_arr[tr_q_idx])
        val_q_set = set(cv_q_arr[val_q_idx])

        fold_tr  = df_cv[df_cv['abs_q'].isin(tr_q_set)]
        fold_val = df_cv[df_cv['abs_q'].isin(val_q_set)]

        tr_label  = f"{abs_to_label(min(tr_q_set))}~{abs_to_label(max(tr_q_set))}"
        val_label = f"{abs_to_label(min(val_q_set))}~{abs_to_label(max(val_q_set))}"
        print(f"  Fold{fold_idx:<3} {tr_label:<22} {val_label:<22} {len(fold_tr)}/{len(fold_val)}")

        X_tr  = fold_tr[X_COLS].values;  y_tr  = fold_tr['Y'].values
        X_val = fold_val[X_COLS].values; y_val = fold_val['Y'].values

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)

        for name, model in get_models().items():
            is_ridge = (name == 'Ridge')
            if is_ridge:
                model.fit(X_tr_sc, y_tr)
                pred     = model.predict(X_val_sc)
                X_pi     = pd.DataFrame(X_val_sc, columns=X_COLS)
            else:
                model.fit(X_tr, y_tr)
                pred     = model.predict(X_val)
                X_pi     = pd.DataFrame(X_val, columns=X_COLS)

            m = evaluate_all(y_val, pred, name, f'CV_Fold{fold_idx}')
            m['h']    = h
            m['fold'] = fold_idx
            fold_metrics[name].append(m)
            all_cv_summary.append(m)

            # PI
            perm = permutation_importance(
                model, X_pi, y_val, n_repeats=5, random_state=42, n_jobs=-1
            )
            fold_pi[name].append(perm.importances_mean)

    # ── CV 평균 성능 출력 ─────────────────────────────────────────
    print(f"\n  [CV 평균 성능 (4 Fold 평균) — 모델 선택 기준]")
    print(f"  {'모델':<12} {'Spearman':>10} {'Top_K_Hit':>10} {'Dir_Acc':>10} {'RMSE':>10} {'R²':>8}")
    print(f"  {'-'*60}")

    cv_mean_rows = []
    for name in get_models():
        folds = fold_metrics[name]
        row = {
            'model':     name,
            'Spearman':  round(np.mean([f['Spearman']  for f in folds]), 4),
            'Top_K_Hit': round(np.mean([f['Top_K_Hit'] for f in folds]), 4),
            'Dir_Acc':   round(np.mean([f['Dir_Acc']   for f in folds]), 4),
            'RMSE':      round(np.mean([f['RMSE']       for f in folds]), 4),
            'R2':        round(np.mean([f['R2']         for f in folds]), 4),
        }
        cv_mean_rows.append(row)
        print(f"  {name:<12} {row['Spearman']:>10.4f} {row['Top_K_Hit']:>10.4f} "
              f"{row['Dir_Acc']:>10.4f} {row['RMSE']:>10.4f} {row['R2']:>8.4f}")

    best_model_name = max(cv_mean_rows, key=lambda x: x['Spearman'])['model']
    print(f"\n  ✅ CV 기준 최고 모델: {best_model_name}")

    # ── 앙상블 PI (fold별 × 모델별 평균) ─────────────────────────
    # 각 모델의 fold 평균 PI → 5개 모델 평균
    model_mean_pi = {}
    for name in get_models():
        model_mean_pi[name] = np.mean(fold_pi[name], axis=0)

    pi_ensemble = np.mean(list(model_mean_pi.values()), axis=0)
    pi_df = pd.DataFrame({
        '변수':       X_COLS,
        'importance': pi_ensemble,
        'log변환':    ['O' if c in log_applied else '' for c in X_COLS]
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    for name, pi_vals in model_mean_pi.items():
        pi_df[f'PI_{name}'] = [dict(zip(X_COLS, pi_vals))[c] for c in pi_df['변수']]

    pi_df.to_csv(os.path.join(OUT, f'h{h}_pi_ensemble.csv'),
                 index=False, encoding='utf-8-sig')

    keep   = pi_df[pi_df['importance'] >= PI_THRESHOLD]['변수'].tolist()
    remove = pi_df[pi_df['importance'] <  PI_THRESHOLD]['변수'].tolist()
    print(f"\n  [앙상블 PI — CV 4 Fold 평균]")
    print(f"  ✅ 유지 ({len(keep)}개): {keep}")
    print(f"  ❌ 제거 ({len(remove)}개): {remove}")
    print(f"  1순위 변수: {pi_df.iloc[0]['변수']} (importance={pi_df.iloc[0]['importance']:.4f})")

    # ── PI 시각화 ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(X_COLS) * 0.45)))

    colors = ['#2196F3' if v >= PI_THRESHOLD else '#BDBDBD' for v in pi_df['importance']]
    axes[0].barh(pi_df['변수'], pi_df['importance'], color=colors)
    axes[0].invert_yaxis()
    axes[0].axvline(x=PI_THRESHOLD, color='red', linestyle='--', label=f'임계값={PI_THRESHOLD}')
    for i, (_, row) in enumerate(pi_df.iterrows()):
        if row['log변환'] == 'O':
            axes[0].text(row['importance'] + 0.0005, i, '[log]', va='center', fontsize=7, color='gray')
    axes[0].legend()
    axes[0].set_title(f'h{h} 앙상블 PI (CV {N_SPLITS} Fold 평균, 5개 모델 평균)')

    top10 = pi_df['변수'].head(10).tolist()
    pi_model_cols = [c for c in pi_df.columns if c.startswith('PI_')]
    pi_top10 = pi_df[pi_df['변수'].isin(top10)].set_index('변수')[pi_model_cols]
    pi_top10.columns = [c.replace('PI_', '') for c in pi_model_cols]
    pi_top10.plot(kind='barh', ax=axes[1], colormap='tab10')
    axes[1].invert_yaxis()
    axes[1].set_title(f'h{h} 모델별 PI 비교 (상위 10개 변수)')
    axes[1].legend(title='모델', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f'h{h}_pi_ensemble.png'), dpi=150)
    plt.close()

    # ── 최종 Test 평가 (딱 한 번) ────────────────────────────────
    # CV 전체 데이터로 재학습 → Test로 최종 평가
    if len(df_test) == 0:
        print(f"\n  ⚠️  Test 데이터 없음, 최종 평가 스킵")
        continue

    print(f"\n  [최종 Test 평가 — CV 선택 모델로 딱 한 번]")
    print(f"  {'모델':<12} {'Spearman':>10} {'Top_K_Hit':>10} {'Dir_Acc':>10} {'RMSE':>10} {'R²':>8}")
    print(f"  {'-'*60}")

    X_cv_all = df_cv[X_COLS].values;  y_cv_all = df_cv['Y'].values
    X_te     = df_test[X_COLS].values; y_te    = df_test['Y'].values

    scaler_final = StandardScaler()
    X_cv_sc = scaler_final.fit_transform(X_cv_all)
    X_te_sc = scaler_final.transform(X_te)

    for name, model in get_models().items():
        is_ridge = (name == 'Ridge')
        if is_ridge:
            model.fit(X_cv_sc, y_cv_all)
            pred_te = model.predict(X_te_sc)
        else:
            model.fit(X_cv_all, y_cv_all)
            pred_te = model.predict(X_te)

        tm = evaluate_all(y_te, pred_te, name, 'Test')
        tm['h'] = h
        all_test_summary.append(tm)

        marker = ' ← CV 최고' if name == best_model_name else ''
        print(f"  {name:<12} {tm['Spearman']:>10.4f} {tm['Top_K_Hit']:>10.4f} "
              f"{tm['Dir_Acc']:>10.4f} {tm['RMSE']:>10.4f} {tm['R2']:>8.4f}{marker}")

# ══════════════════════════════════════════════════════════════════
# 전체 요약 저장 + 시각화
# ══════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print("전체 요약")
print(f"{'═'*70}")

cv_df   = pd.DataFrame(all_cv_summary)
test_df = pd.DataFrame(all_test_summary) if all_test_summary else pd.DataFrame()

# CV fold별 raw 저장
cv_df.to_csv(os.path.join(OUT, 'summary_cv_folds.csv'), index=False, encoding='utf-8-sig')

# CV 평균 저장
cv_mean_df = (cv_df.groupby(['h','model'])[['Spearman','Top_K_Hit','Dir_Acc','RMSE','R2']]
              .mean().round(4).reset_index())
cv_mean_df['split'] = 'CV_평균'
cv_mean_df.to_csv(os.path.join(OUT, 'summary_cv_mean.csv'), index=False, encoding='utf-8-sig')

if not test_df.empty:
    test_df.to_csv(os.path.join(OUT, 'summary_test.csv'), index=False, encoding='utf-8-sig')

# 콘솔 출력
print("\n[CV 평균 Spearman]")
print(cv_mean_df.pivot_table(index='model', columns='h', values='Spearman').round(4).to_string())
if not test_df.empty:
    print("\n[Test Spearman]")
    print(test_df.pivot_table(index='model', columns='h', values='Spearman').round(4).to_string())
    print("\n[Test R²]")
    print(test_df.pivot_table(index='model', columns='h', values='R2').round(4).to_string())

# ── CV 평균 vs Test Spearman 꺾은선 비교 ─────────────────────────
if not test_df.empty:
    model_names = list(get_models().keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(20, 4))
    fig.suptitle('모델별  CV 평균 vs Test  Spearman (h별)\n'
                 '※ CV와 Test 차이가 클수록 과적합 의심', fontsize=12)

    for ax, mname in zip(axes, model_names):
        cv_m  = cv_mean_df[cv_mean_df['model'] == mname].sort_values('h')
        te_m  = test_df[test_df['model'] == mname].sort_values('h')
        ax.plot(cv_m['h'], cv_m['Spearman'], 'o-', label='CV 평균', color='steelblue', linewidth=2)
        ax.plot(te_m['h'], te_m['Spearman'], 's--', label='Test',    color='tomato',    linewidth=2)
        ax.fill_between(cv_m['h'], cv_m['Spearman'], te_m['Spearman'],
                        alpha=0.1, color='gray', label='차이')
        ax.set_title(mname, fontsize=10)
        ax.set_xlabel('h (예측 분기)')
        ax.set_xticks([1, 2, 3, 4])
        ax.set_ylim(-0.2, 1.0)
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'summary_cv_vs_test_spearman.png'), dpi=150)
    plt.close()

    # ── 4개 지표 막대 비교 ────────────────────────────────────────
    metrics_list = ['Spearman', 'Top_K_Hit', 'Dir_Acc', 'R2']
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle('7트_시뮬_1 v3  CV 평균 vs Test 성능 비교', fontsize=14)

    for col_i, metric in enumerate(metrics_list):
        for row_i, (label, df_s) in enumerate([('CV 평균', cv_mean_df), ('Test', test_df)]):
            ax = axes[row_i][col_i]
            pivot = df_s.pivot_table(index='model', columns='h', values=metric)
            pivot.plot(kind='bar', ax=ax, rot=30, colormap='tab10',
                       legend=(row_i == 0 and col_i == 3))
            ax.set_title(f'[{label}] {metric}', fontsize=10)
            ax.set_xlabel('')
            ax.grid(axis='y', alpha=0.3)
            if row_i == 0 and col_i == 3:
                ax.legend(title='h', loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'summary_cv_vs_test_전체비교.png'), dpi=150)
    plt.close()

print(f"\n===== 7트_시뮬_1 STEP2 v3 완료 =====")
print(f"저장 위치: {OUT}")
print("""
  출력 파일 목록
  ───────────────────────────────────────────────────────────
  summary_cv_folds.csv              ← CV fold별 raw 성능
  summary_cv_mean.csv               ← CV fold 평균 성능 (모델 선택 기준)
  summary_test.csv                  ← 최종 Test 성능 (보고 기준)
  summary_cv_vs_test_spearman.png   ← CV vs Test Spearman 꺾은선
  summary_cv_vs_test_전체비교.png   ← 4개 지표 CV vs Test 막대 비교
  h{h}_pi_ensemble.csv              ← 앙상블 PI (CV fold 평균)
  h{h}_pi_ensemble.png              ← PI 시각화 + 모델별 비교
  ───────────────────────────────────────────────────────────
""")