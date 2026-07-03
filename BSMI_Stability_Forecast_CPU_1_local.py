#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BSMI 大氣穩定度預測系統 v2 — 時間序列分析版
Atmospheric Stability Forecast via Time Series Analysis

核心改進（vs v1）：
  1. 時間序列特徵萃取 — 多尺度滾動統計、差分、EWMA、趨勢、頻域特徵
  2. Horizon-as-Feature — 單一模型可預測任意未來時間點（1–120 分鐘）
  3. Expanding Window CV — 嚴格時序交叉驗證 + embargo 防洩漏
  4. 連續預報軌跡 — 1 分鐘解析度的未來 2 小時預報 + 穩定度轉換預警

本版本在本地環境執行（無需 Google Colab / Google Drive）。
"""

# ============================================================
# Cell 1: 安裝相依套件
# ============================================================
import subprocess, sys

def install_packages():
    """安裝必要套件（若尚未安裝）"""
    packages = [
        'xgboost', 'lightgbm', 'pyarrow', 'pandas',
        'scikit-learn', 'joblib', 'matplotlib', 'seaborn'
    ]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f'Installing {pkg}...')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

install_packages()

# ============================================================
# Cell 2: (已移除 Google Colab drive.mount — 本地不需要)
# ============================================================

# ============================================================
# Cell 3: 匯入套件 & 設定參數
# ============================================================
import numpy as np, pandas as pd, math, warnings, glob, os, joblib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC','Microsoft JhengHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---- 本地路徑設定 ----
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = os.path.join(SCRIPT_DIR, 'BSMI_wind_1min_parquet')
MODEL_DIR   = os.path.join(SCRIPT_DIR, 'models_stability_v2')
os.makedirs(MODEL_DIR, exist_ok=True)

HEIGHTS    = np.array([38.0, 69.0, 100.0])
LN_HEIGHTS = np.log(HEIGHTS)
STABLE_THRESHOLD = 0.20   # binary: alpha >= 0.20 → stable

# ---- 時間序列參數 ----
WINDOW       = 120                                       # 特徵暖機長度（分鐘）
FORECAST_MAX = 120                                       # 最大預報距離（分鐘）
TRAIN_HORIZONS = [1,2,3,5,10,15,20,30,45,60,90,120]     # 訓練用 horizon 取樣
STRIDE       = 5                                         # 取樣步長
GAP_MIN      = 1.5                                       # 斷段閾值（分鐘）
N_CV_SPLITS  = 5                                         # Expanding Window CV 折數
FFT_WINDOW   = 60                                        # FFT 特徵窗口

LGBM_PARAMS = dict(n_estimators=1000, learning_rate=0.08, max_depth=14,
                    num_leaves=256, subsample=0.8, colsample_bytree=0.8,
                    min_child_samples=50, random_state=42, verbose=-1)
EARLY_STOP = 50

print(f'Window={WINDOW}, ForecastMax={FORECAST_MAX}min, Horizons={TRAIN_HORIZONS}')
print(f'Stride={STRIDE}, CV_Splits={N_CV_SPLITS}')

# ============================================================
# Cell 4: 載入 Parquet 資料
# ============================================================
files = sorted(glob.glob(os.path.join(PARQUET_DIR, '*.parquet')))
print(f'\nFound {len(files)} parquet files')
if len(files) == 0:
    raise FileNotFoundError(
        f'在 {PARQUET_DIR} 中找不到任何 .parquet 檔案。\n'
        f'請將 Parquet 資料放在此目錄下再執行。'
    )
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=False).sort_index()
print(f'Total {len(df):,} rows, {df.index.min()} ~ {df.index.max()}')

# ============================================================
# Cell 5: 時間序列特徵工程
# ============================================================
def compute_alpha(df):
    """利用 3 高度風速剖面計算冪律指數 alpha"""
    ws100 = (df['WS_100E'] + df['WS_100W']) / 2
    ws69, ws38 = df['WS_69W'], df['WS_38W']
    WS_mat = np.column_stack([ws38.values, ws69.values, ws100.values])
    valid = (WS_mat > 0.1).all(axis=1)
    LN_WS = np.log(np.clip(WS_mat, 0.1, None))
    lnz_mean = LN_HEIGHTS.mean()
    lnws_mean = LN_WS.mean(axis=1)
    numer = np.zeros(len(df)); denom = 0.0
    for i in range(3):
        numer += (LN_HEIGHTS[i] - lnz_mean) * (LN_WS[:, i] - lnws_mean)
        denom += (LN_HEIGHTS[i] - lnz_mean) ** 2
    alpha = numer / denom
    alpha[~valid] = np.nan
    return alpha


def compute_fft_features(alpha_series, window=FFT_WINDOW, chunk_size=80000):
    """計算滾動 FFT 頻域特徵：主週期 & 低頻能量比"""
    vals = alpha_series.values.astype(np.float64)
    n = len(vals)
    dom_period    = np.full(n, np.nan, dtype=np.float32)
    spectral_ratio = np.full(n, np.nan, dtype=np.float32)
    freqs = np.fft.rfftfreq(window, d=1.0)[1:]   # skip DC
    low_mask = freqs < (1.0 / 20)                 # periods > 20 min

    for cs in range(window - 1, n, chunk_size):
        ce = min(cs + chunk_size, n)
        indices = np.arange(cs, ce)
        # 建構批次窗口矩陣
        batch = np.array([vals[i - window + 1:i + 1] for i in indices],
                         dtype=np.float64)
        nan_rows = np.any(np.isnan(batch), axis=1)
        batch_c = batch.copy()
        batch_c[nan_rows] = 0
        batch_c -= batch_c.mean(axis=1, keepdims=True)   # 去趨勢
        fft_vals = np.abs(np.fft.rfft(batch_c, axis=1))[:, 1:]

        peak_idx = np.argmax(fft_vals, axis=1)
        dp = np.where(freqs[peak_idx] > 0, 1.0 / freqs[peak_idx], np.nan)
        low_e  = fft_vals[:, low_mask].sum(axis=1)
        total_e = fft_vals.sum(axis=1)
        sr = np.where(total_e > 0, low_e / total_e, np.nan)
        dp[nan_rows] = np.nan
        sr[nan_rows] = np.nan

        dom_period[indices]     = dp.astype(np.float32)
        spectral_ratio[indices] = sr.astype(np.float32)
    return dom_period, spectral_ratio


def add_ts_features(df):
    """
    時間序列特徵工程 —— 從原始 1 分鐘資料萃取統計摘要特徵。
    每一列 = 一個時間步的完整狀態描述（~71 維），取代舊方法的壓平原始值。
    """
    df = df.copy()
    ws100 = (df['WS_100E'] + df['WS_100W']) / 2
    ws69  = df['WS_69W']
    ws38  = df['WS_38W']

    # ---- Alpha 計算 ----
    df['alpha'] = compute_alpha(df)
    alpha = df['alpha']

    # ---- A. 多尺度滾動統計 ----
    for w in [5, 15, 30, 60, 120]:
        mp = max(w // 2, 3)
        df[f'alpha_mean_{w}'] = alpha.rolling(w, min_periods=mp).mean()
        df[f'alpha_std_{w}']  = alpha.rolling(w, min_periods=mp).std()
    for w in [30, 60, 120]:
        mp = max(w // 2, 3)
        df[f'alpha_min_{w}'] = alpha.rolling(w, min_periods=mp).min()
        df[f'alpha_max_{w}'] = alpha.rolling(w, min_periods=mp).max()
    for w in [30, 60]:
        mp = max(w // 2, 3)
        df[f'alpha_skew_{w}'] = alpha.rolling(w, min_periods=mp).skew()

    # ---- B. 差分特徵 ----
    df['alpha_diff_1']  = alpha.diff(1)
    df['alpha_diff2']   = alpha.diff(1).diff(1)        # 二階差分（加速度）
    df['alpha_diff_5']  = alpha.diff(5)
    df['alpha_diff_15'] = alpha.diff(15)
    df['alpha_diff_30'] = alpha.diff(30)
    df['alpha_diff_60'] = alpha.diff(60)

    # ---- C. 指數加權移動平均 ----
    for span in [10, 30, 60]:
        df[f'alpha_ewm_{span}'] = alpha.ewm(span=span).mean()

    # ---- D. 滯後特徵（自相關代理） ----
    for lag in [5, 15, 30, 60]:
        df[f'alpha_lag_{lag}'] = alpha.shift(lag)

    # ---- E. 趨勢斜率 ----
    for w in [30, 60]:
        df[f'alpha_trend_{w}'] = (alpha - alpha.shift(w)) / w

    # ---- F. FFT 頻域特徵 ----
    print('  Computing FFT spectral features...')
    dom_p, spec_r = compute_fft_features(alpha, FFT_WINDOW)
    df['alpha_dom_period']     = dom_p
    df['alpha_spectral_ratio'] = spec_r

    # ---- 風切特徵（多尺度） ----
    shear = ws100 - ws38
    df['shear_current'] = shear
    for w in [15, 30, 60]:
        mp = max(w // 2, 3)
        df[f'shear_mean_{w}'] = shear.rolling(w, min_periods=mp).mean()
        df[f'shear_std_{w}']  = shear.rolling(w, min_periods=mp).std()
    df['shear_diff_30'] = shear.diff(30)

    # ---- 氣象特徵 ----
    df['BP_diff_30'] = df['BP_93'].diff(30)
    df['BP_diff_60'] = df['BP_93'].diff(60)
    df['BP_std_60']  = df['BP_93'].rolling(60, min_periods=30).std()
    df['AT_diff_30'] = df['AT_95'].diff(30)
    df['AT_diff_60'] = df['AT_95'].diff(60)
    df['RH_diff_60'] = df['RH_95'].diff(60)

    # 紊流強度
    m100 = ws100.rolling(10, min_periods=5).mean()
    s100 = ws100.rolling(10, min_periods=5).std()
    m38  = ws38.rolling(10, min_periods=5).mean()
    s38  = ws38.rolling(10, min_periods=5).std()
    df['TI_100']   = s100 / (m100 + 0.01)
    df['TI_38']    = s38  / (m38  + 0.01)
    df['TI_ratio'] = df['TI_100'] / (df['TI_38'] + 0.01)

    # 露點溫度差
    RH = np.clip(df['RH_95'], 1, 100)
    g = (17.27 * df['AT_95']) / (237.7 + df['AT_95']) + np.log(RH / 100)
    df['Td_95']     = (237.7 * g) / (17.27 - g)
    df['T_Td_diff'] = df['AT_95'] - df['Td_95']

    # 當前風速均值
    df['WS_mean'] = (df['WS_100E'] + df['WS_100W'] + ws69 + ws38) / 4

    # ---- 時間編碼 ----
    minute_of_day = df.index.hour * 60 + df.index.minute
    df['hour_sin']   = np.sin(2 * math.pi * df.index.hour / 24)
    df['hour_cos']   = np.cos(2 * math.pi * df.index.hour / 24)
    df['month_sin']  = np.sin(2 * math.pi * df.index.month / 12)
    df['month_cos']  = np.cos(2 * math.pi * df.index.month / 12)
    df['minute_sin'] = np.sin(2 * math.pi * minute_of_day / 1440)
    df['minute_cos'] = np.cos(2 * math.pi * minute_of_day / 1440)

    # ---- 平滑化目標 ----
    df['alpha_30'] = alpha.rolling(30, min_periods=15).mean()

    return df


print('Computing time series features...')
df = add_ts_features(df)

# ---- 定義特徵列名（共 71 維） ----
ALPHA_ROLL = ([f'alpha_mean_{w}' for w in [5,15,30,60,120]] +
              [f'alpha_std_{w}'  for w in [5,15,30,60,120]] +
              [f'alpha_min_{w}'  for w in [30,60,120]] +
              [f'alpha_max_{w}'  for w in [30,60,120]] +
              [f'alpha_skew_{w}' for w in [30,60]])

ALPHA_DIFF = ['alpha_diff_1','alpha_diff2','alpha_diff_5',
              'alpha_diff_15','alpha_diff_30','alpha_diff_60']

ALPHA_EWMA  = [f'alpha_ewm_{s}' for s in [10,30,60]]
ALPHA_LAG   = [f'alpha_lag_{l}' for l in [5,15,30,60]]
ALPHA_TREND = [f'alpha_trend_{w}' for w in [30,60]]
ALPHA_FFT   = ['alpha_dom_period','alpha_spectral_ratio']

SHEAR_FEATS = (['shear_current'] +
               [f'shear_{s}_{w}' for s in ['mean','std'] for w in [15,30,60]] +
               ['shear_diff_30'])

METEO_FEATS = ['BP_93','AT_95','RH_95',
               'BP_diff_30','BP_diff_60','BP_std_60',
               'AT_diff_30','AT_diff_60','RH_diff_60',
               'TI_100','TI_38','TI_ratio','T_Td_diff']

WIND_FEATS = ['WS_100E','WS_100W','WS_69W','WS_38W','WS_mean']
WD_FEATS   = ['WD_97_sin','WD_97_cos','WD_35_sin','WD_35_cos']
TIME_FEATS = ['hour_sin','hour_cos','month_sin','month_cos','minute_sin','minute_cos']

TS_FEAT_COLS = (ALPHA_ROLL + ALPHA_DIFF + ALPHA_EWMA + ALPHA_LAG +
                ALPHA_TREND + ALPHA_FFT + SHEAR_FEATS + METEO_FEATS +
                WIND_FEATS + WD_FEATS + TIME_FEATS)

df = df.dropna(subset=TS_FEAT_COLS + ['alpha_30'])
print(f'After TS features: {len(df):,} rows, {len(TS_FEAT_COLS)} features')
print(f'  Alpha-roll:{len(ALPHA_ROLL)} Diff:{len(ALPHA_DIFF)} EWMA:{len(ALPHA_EWMA)} '
      f'Lag:{len(ALPHA_LAG)} Trend:{len(ALPHA_TREND)} FFT:{len(ALPHA_FFT)}')
print(f'  Shear:{len(SHEAR_FEATS)} Meteo:{len(METEO_FEATS)} '
      f'Wind:{len(WIND_FEATS)} WD:{len(WD_FEATS)} Time:{len(TIME_FEATS)}')

# ============================================================
# Cell 6: 繪製穩定度分佈圖
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
a = df['alpha'].dropna()

ax = axes[0]
ax.hist(a.clip(-1, 1.5), bins=80, color='steelblue', alpha=0.8, edgecolor='white')
for thr, col, lbl in [(0.10,'orange','unstable/neutral'), (0.20,'red','neutral/stable')]:
    ax.axvline(thr, color=col, ls='--', lw=2, label=f'α={thr}')
ax.set_xlabel('alpha'); ax.set_ylabel('Count')
ax.set_title('Wind Profile Power Law Exponent α'); ax.legend(fontsize=8)

ax = axes[1]
hourly = df.groupby(df.index.hour)['alpha'].agg(['mean','std'])
ax.fill_between(hourly.index, hourly['mean']-hourly['std'],
                hourly['mean']+hourly['std'], alpha=0.2, color='steelblue')
ax.plot(hourly.index, hourly['mean'], 'o-', color='steelblue', lw=2)
ax.axhline(0.10, color='orange', ls=':', lw=1)
ax.axhline(0.20, color='red', ls=':', lw=1)
ax.set_xlabel('Hour'); ax.set_ylabel('alpha')
ax.set_title('Diurnal Pattern (night>day = correct)'); ax.set_xticks(range(0, 24, 3))

ax = axes[2]
n_stable = (a >= STABLE_THRESHOLD).sum()
n_notstable = (a < STABLE_THRESHOLD).sum()
ax.pie([n_notstable, n_stable],
       labels=['Not-Stable\n(α<0.20)', 'Stable\n(α≥0.20)'],
       colors=['#5BA3E6','#FF8C42'], autopct='%1.1f%%', startangle=90)
ax.set_title('Stability Class Distribution')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'stability_distribution.png'), dpi=120)
plt.show()

night = df.loc[df.index.hour.isin([20,21,22,23,0,1,2,3,4]), 'alpha'].mean()
day   = df.loc[df.index.hour.isin([10,11,12,13,14,15,16]), 'alpha'].mean()
print(f'Night α={night:.3f}  Day α={day:.3f}  diff={night-day:.3f}  '
      f'(night>day = physics correct)')

# ============================================================
# Cell 7: 建構 Horizon-as-Feature 樣本
# ============================================================
def find_segments(df, gap_min=GAP_MIN):
    """找出連續不中斷的資料段（回傳各段的 iloc 位置陣列）"""
    idx = df.index
    gaps = np.where(np.diff(idx.asi8) > int(gap_min * 60 * 1e9))[0] + 1
    return np.split(np.arange(len(df)), gaps)


def build_haf_samples(df, segments, horizons=TRAIN_HORIZONS, stride=STRIDE):
    """
    Horizon-as-Feature 樣本建構。
    每個樣本 = [TS特徵@t, horizon_h]  →  alpha_30(t+h)
    回傳: X, y_reg, y_cls, source_positions
    """
    feat_arr   = df[TS_FEAT_COLS].values.astype(np.float32)
    alpha_tgt  = df['alpha_30'].values.astype(np.float32)
    max_h = max(horizons)
    F = len(TS_FEAT_COLS)

    # 預估樣本數
    total = 0
    valid_segs = []
    for s in segments:
        usable = len(s) - max_h
        if usable <= 0:
            continue
        total += ((usable + stride - 1) // stride) * len(horizons)
        valid_segs.append(s)

    X    = np.empty((total, F + 1), dtype=np.float32)   # +1 = horizon
    y    = np.empty(total, dtype=np.float32)
    tpos = np.empty(total, dtype=np.int64)               # source position
    k = 0

    for s in valid_segs:
        usable = len(s) - max_h
        for i in range(0, usable, stride):
            src = s[i]
            if not np.all(np.isfinite(feat_arr[src])):
                continue
            for h in horizons:
                tgt = s[i + h]
                tv  = alpha_tgt[tgt]
                if not np.isfinite(tv):
                    continue
                X[k, :F] = feat_arr[src]
                X[k,  F] = float(h)
                y[k]     = tv
                tpos[k]  = src
                k += 1

    X, y, tpos = X[:k], y[:k], tpos[:k]
    y_c = (y >= STABLE_THRESHOLD).astype(np.int32)
    print(f'Built {k:,} HaF samples  '
          f'({len(valid_segs)} segments × {len(horizons)} horizons, stride={stride})')
    return X, y, y_c, tpos


segments = find_segments(df)
print(f'\nFound {len(segments)} continuous segments')
X_all, y_reg_all, y_cls_all, src_pos = build_haf_samples(df, segments)

# ============================================================
# Cell 8: 定義訓練 & 評估函式（Expanding Window CV）
# ============================================================
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, f1_score, classification_report)
import lightgbm as lgb


def train_expanding_cv(X, y_r, y_c, src_pos):
    """
    Expanding Window 交叉驗證 + 最終模型訓練。
    含 embargo（= FORECAST_MAX 分鐘）防止 target 洩漏。
    """
    # 按 source 時間排序
    order = np.argsort(src_pos)
    X, y_r, y_c, src_pos = X[order], y_r[order], y_c[order], src_pos[order]

    n = len(X)
    # 計算 embargo 的樣本數（每分鐘約有 len(TRAIN_HORIZONS)/STRIDE 個樣本）
    time_span = max(src_pos[-1] - src_pos[0], 1)
    samples_per_pos = n / time_span
    embargo_samples = int(FORECAST_MAX * samples_per_pos) + 1
    block_size = n // (N_CV_SPLITS + 1)

    print(f'\n{"="*60}')
    print(f' Expanding Window CV  ({N_CV_SPLITS} folds, embargo={embargo_samples} samples)')
    print(f'{"="*60}')

    all_fold_metrics = []
    last_fold_test_idx = None

    for fold in range(N_CV_SPLITS):
        train_end  = (fold + 1) * block_size
        test_start = train_end + embargo_samples
        test_end   = min((fold + 2) * block_size, n)

        if test_start >= test_end:
            print(f'  Fold {fold+1}: skipped (embargo too large)')
            continue

        # 訓練集再拆出驗證集（最後 10%，但保留 embargo）
        val_size  = max(int(train_end * 0.10), 200)
        val_start = train_end - val_size
        tr_end    = val_start

        tr_idx  = np.arange(0, tr_end)
        va_idx  = np.arange(val_start, train_end)
        te_idx  = np.arange(test_start, test_end)

        print(f'\n  Fold {fold+1}/{N_CV_SPLITS}: '
              f'train={len(tr_idx):,}  val={len(va_idx):,}  test={len(te_idx):,}')

        # Regression
        reg = LGBMRegressor(**LGBM_PARAMS)
        reg.fit(X[tr_idx], y_r[tr_idx],
                eval_set=[(X[va_idx], y_r[va_idx])],
                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                           lgb.log_evaluation(0)])
        print(f'    Reg best_iter={reg.best_iteration_}')

        # Classification
        clf = LGBMClassifier(**LGBM_PARAMS, objective='binary')
        clf.fit(X[tr_idx], y_c[tr_idx],
                eval_set=[(X[va_idx], y_c[va_idx])],
                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                           lgb.log_evaluation(0)])
        print(f'    Clf best_iter={clf.best_iteration_}')

        # 評估
        yp  = reg.predict(X[te_idx])
        ycp = clf.predict(X[te_idx])
        yt  = y_r[te_idx]
        yct = y_c[te_idx]

        m = dict(fold=fold+1,
                 RMSE=np.sqrt(mean_squared_error(yt, yp)),
                 MAE=mean_absolute_error(yt, yp),
                 R2=r2_score(yt, yp),
                 Accuracy=accuracy_score(yct, ycp),
                 Macro_F1=f1_score(yct, ycp, average='macro'))
        all_fold_metrics.append(m)
        print(f'    RMSE={m["RMSE"]:.4f}  MAE={m["MAE"]:.4f}  '
              f'R²={m["R2"]:.4f}  Acc={m["Accuracy"]:.3f}  F1={m["Macro_F1"]:.3f}')

        last_fold_test_idx = te_idx

    # ---- 最終模型：使用全部資料訓練 ----
    print(f'\n  Training FINAL model on all {n:,} samples...')
    val_cut = int(n * 0.90)
    final_reg = LGBMRegressor(**LGBM_PARAMS)
    final_reg.fit(X[:val_cut], y_r[:val_cut],
                  eval_set=[(X[val_cut:], y_r[val_cut:])],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                             lgb.log_evaluation(0)])
    print(f'    Final Reg best_iter={final_reg.best_iteration_}')

    final_clf = LGBMClassifier(**LGBM_PARAMS, objective='binary')
    final_clf.fit(X[:val_cut], y_c[:val_cut],
                  eval_set=[(X[val_cut:], y_c[val_cut:])],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                             lgb.log_evaluation(0)])
    print(f'    Final Clf best_iter={final_clf.best_iteration_}')

    models = {'reg': final_reg, 'clf': final_clf}
    return models, all_fold_metrics, last_fold_test_idx, X, y_r, y_c, src_pos

# ============================================================
# Cell 9: 執行訓練 & 儲存
# ============================================================
(trained_models, cv_metrics, test_idx,
 X_sorted, y_reg_sorted, y_cls_sorted, src_sorted) = train_expanding_cv(
    X_all, y_reg_all, y_cls_all, src_pos)

# 儲存模型
joblib.dump(trained_models, os.path.join(MODEL_DIR, 'STAB_HaF_final.pkl'))
print(f'\n  Saved STAB_HaF_final.pkl')

# CV 報告
report = pd.DataFrame(cv_metrics)
report.to_csv(os.path.join(MODEL_DIR, 'cv_evaluation_report.csv'), index=False)
print('\n=== Cross-Validation Summary ===')
print(report.to_string(index=False))
if len(cv_metrics) > 1:
    for col in ['RMSE','MAE','R2','Accuracy','Macro_F1']:
        vals = report[col]
        print(f'  {col}: {vals.mean():.4f} ± {vals.std():.4f}')

# ============================================================
# Cell 10: 預報技巧衰減曲線（Forecast Skill vs Horizon）
# ============================================================
print('\n=== Forecast Skill Decay ===')
if test_idx is not None:
    X_te = X_sorted[test_idx]
    y_te = y_reg_sorted[test_idx]
    yc_te = y_cls_sorted[test_idx]
    horizons_te = X_te[:, -1]   # last column = horizon

    # 按 horizon 分群評估
    skill_data = []
    for h in TRAIN_HORIZONS:
        mask = horizons_te == h
        if mask.sum() < 30:
            continue
        yp = trained_models['reg'].predict(X_te[mask])
        ycp = trained_models['clf'].predict(X_te[mask])
        yt_h = y_te[mask]
        yct_h = yc_te[mask]
        skill_data.append(dict(
            horizon=h,
            RMSE=np.sqrt(mean_squared_error(yt_h, yp)),
            MAE=mean_absolute_error(yt_h, yp),
            R2=r2_score(yt_h, yp),
            Accuracy=accuracy_score(yct_h, ycp),
            n=int(mask.sum())
        ))

    skill_df = pd.DataFrame(skill_data)
    print(skill_df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    hs = skill_df['horizon']

    ax = axes[0]
    ax.plot(hs, skill_df['RMSE'], 'o-', color='#E74C3C', lw=2, markersize=6)
    ax.set_xlabel('Forecast Horizon (min)'); ax.set_ylabel('RMSE')
    ax.set_title('RMSE vs Horizon (↑ = worse)'); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(hs, skill_df['R2'], 's-', color='#2E86C1', lw=2, markersize=6)
    ax.set_xlabel('Forecast Horizon (min)'); ax.set_ylabel('R²')
    ax.set_title('R² vs Horizon (↓ = worse)'); ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', ls=':', lw=1)

    ax = axes[2]
    ax.plot(hs, skill_df['Accuracy'], 'D-', color='#27AE60', lw=2, markersize=6)
    ax.set_xlabel('Forecast Horizon (min)'); ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy vs Horizon'); ax.grid(alpha=0.3)

    plt.suptitle('Forecast Skill Decay — How accuracy degrades with forecast distance',
                 fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'forecast_skill_decay.png'), dpi=120)
    plt.show()

# ============================================================
# Cell 11: 連續預報軌跡圖
# ============================================================
print('\n=== Continuous Forecast Trajectories ===')

def predict_trajectory(df, model, start_pos, max_horizon=FORECAST_MAX):
    """從 start_pos 出發，預測未來 1~max_horizon 分鐘的 alpha 軌跡"""
    feat = df.iloc[start_pos][TS_FEAT_COLS].values.astype(np.float32)
    if not np.all(np.isfinite(feat)):
        return None, None
    pred = np.empty(max_horizon, dtype=np.float32)
    for h in range(1, max_horizon + 1):
        x = np.append(feat, float(h)).reshape(1, -1)
        pred[h - 1] = model.predict(x)[0]
    # 實際值
    actual = np.full(max_horizon, np.nan, dtype=np.float32)
    alpha_vals = df['alpha_30'].values
    for h in range(1, max_horizon + 1):
        pos = start_pos + h
        if pos < len(df):
            actual[h - 1] = alpha_vals[pos]
    return pred, actual


# 在測試期間找出可用的軌跡起點
if test_idx is not None:
    test_positions = np.unique(src_sorted[test_idx])
    # 只保留後續有完整 FORECAST_MAX 分鐘資料的起點
    seg_sets = [set(s) for s in segments]
    valid_starts = []
    for p in test_positions:
        # 確認 p ~ p+FORECAST_MAX 都在同一 segment 內
        for ss in seg_sets:
            if p in ss and p + FORECAST_MAX in ss:
                valid_starts.append(p)
                break
    print(f'  {len(valid_starts)} valid trajectory starting points')

    if len(valid_starts) >= 4:
        # 均勻取 4 個起點
        pick_idx = np.linspace(0, len(valid_starts)-1, 4, dtype=int)
        selected = [valid_starts[i] for i in pick_idx]

        fig, axes = plt.subplots(4, 1, figsize=(14, 14))
        for ax, sp in zip(axes, selected):
            pred, actual = predict_trajectory(df, trained_models['reg'], sp)
            if pred is None:
                continue
            x = np.arange(1, FORECAST_MAX + 1)
            ax.plot(x, actual, 'b-', lw=1.2, label='Actual α', alpha=0.8)
            ax.plot(x, pred, 'r-', lw=1.2, label='Predicted α', alpha=0.8)
            ax.axhline(STABLE_THRESHOLD, color='red', ls='--', lw=1, alpha=0.5,
                       label=f'Threshold={STABLE_THRESHOLD}')
            ax.fill_between(x, STABLE_THRESHOLD, pred,
                            where=(pred >= STABLE_THRESHOLD),
                            color='#FF8C42', alpha=0.15, label='Stable zone')

            # 偵測穩定度轉換
            current_stable = (df['alpha_30'].values[sp] >= STABLE_THRESHOLD)
            for i in range(len(pred)):
                is_stable = pred[i] >= STABLE_THRESHOLD
                if is_stable != current_stable:
                    trans = 'Stable→Not-Stable' if current_stable else 'Not-Stable→Stable'
                    ax.axvline(i+1, color='green', ls=':', lw=1.5, alpha=0.8)
                    ax.annotate(f'⚠ {trans} @ t+{i+1}min',
                               xy=(i+1, STABLE_THRESHOLD), fontsize=8,
                               color='green', fontweight='bold',
                               xytext=(i+10, STABLE_THRESHOLD + 0.05),
                               arrowprops=dict(arrowstyle='->', color='green'))
                    break

            ts = df.index[sp]
            ax.set_title(f'Forecast from {ts}', fontsize=11)
            ax.set_ylabel('α'); ax.legend(fontsize=7, loc='upper right')
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel('Forecast Horizon (minutes)')
        plt.suptitle('Continuous 2-Hour Stability Forecast Trajectories',
                     fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'forecast_trajectories.png'), dpi=120)
        plt.show()
    else:
        print('  Not enough valid starting points for trajectory plot')

# ============================================================
# Cell 12: 混淆矩陣（按 horizon 分組）
# ============================================================
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

if test_idx is not None:
    show_horizons = [h for h in [15, 30, 60, 120] if h in TRAIN_HORIZONS]
    fig, axes = plt.subplots(1, len(show_horizons), figsize=(5*len(show_horizons), 4))
    if len(show_horizons) == 1:
        axes = [axes]
    for ax, h in zip(axes, show_horizons):
        mask = horizons_te == h
        if mask.sum() < 10:
            continue
        ycp = trained_models['clf'].predict(X_te[mask])
        yct = yc_te[mask]
        cm = confusion_matrix(yct, ycp)
        ConfusionMatrixDisplay(cm, display_labels=['Not-Stable','Stable']).plot(
            ax=ax, cmap='Blues', colorbar=False)
        acc = accuracy_score(yct, ycp)
        ax.set_title(f't+{h}min  Acc={acc:.3f}')
    plt.suptitle('Stability Classification Confusion Matrix (by horizon)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'stability_confusion.png'), dpi=120)
    plt.show()

# ============================================================
# Cell 13: 特徵重要性
# ============================================================
reg = trained_models['reg']
feat_names = TS_FEAT_COLS + ['horizon_min']
imp = reg.feature_importances_
top_k = min(25, len(feat_names))
idx_top = np.argsort(imp)[-top_k:]

fig, ax = plt.subplots(figsize=(9, 7))
colors = []
for i in idx_top:
    name = feat_names[i]
    if name == 'horizon_min':
        colors.append('#E74C3C')       # red for horizon
    elif 'alpha' in name:
        colors.append('darkorange')    # orange for alpha TS features
    elif 'shear' in name:
        colors.append('#8E44AD')       # purple for shear
    else:
        colors.append('steelblue')     # default

ax.barh(range(top_k), imp[idx_top], color=colors)
ax.set_yticks(range(top_k))
ax.set_yticklabels([feat_names[i] for i in idx_top], fontsize=8)
ax.set_xlabel('Importance')
ax.set_title(f'Top {top_k} Feature Importances')

# 顏色圖例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='darkorange', label='Alpha TS features'),
    Patch(facecolor='#E74C3C', label='Horizon feature'),
    Patch(facecolor='#8E44AD', label='Shear features'),
    Patch(facecolor='steelblue', label='Other'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'), dpi=120)
plt.show()

# ============================================================
# Cell 14: 推論示範 — 連續預報 + 穩定度轉換預警
# ============================================================
def predict_stability_trajectory(df_recent, models, max_horizon=FORECAST_MAX):
    """
    給定近期資料，產出連續預報軌跡。
    回傳: list of dict，每個 dict = {minute, alpha, cls_name}
    """
    feat_cols = TS_FEAT_COLS
    last_row = df_recent.iloc[-1]
    feat = last_row[feat_cols].values.astype(np.float32)
    if not np.all(np.isfinite(feat)):
        print('Warning: features contain NaN, cannot predict')
        return None

    results = []
    current_alpha = last_row.get('alpha_30', np.nan)
    current_stable = current_alpha >= STABLE_THRESHOLD if np.isfinite(current_alpha) else None

    for h in range(1, max_horizon + 1):
        x = np.append(feat, float(h)).reshape(1, -1)
        ap = models['reg'].predict(x)[0]
        cp = models['clf'].predict(x)[0]
        cn = {0: 'Not-Stable', 1: 'Stable'}
        results.append(dict(minute=h, alpha=float(ap), cls=int(cp), cls_name=cn[cp]))

    return results, current_stable


print('\n=== Inference Demo: Continuous 2-Hour Forecast ===')
last_data = df.iloc[-WINDOW:]
trajectory, current_stable = predict_stability_trajectory(last_data, trained_models)

if trajectory:
    print(f'\n  Current state: {"Stable" if current_stable else "Not-Stable"}')
    print(f'  {"─"*50}')
    print(f'  {"Horizon":>8}  {"α":>8}  {"Class":>12}')
    print(f'  {"─"*50}')

    # 找出轉換點
    transition = None
    for r in trajectory:
        is_stable = r['alpha'] >= STABLE_THRESHOLD
        if current_stable is not None and is_stable != current_stable and transition is None:
            transition = r

    for r in trajectory:
        marker = ''
        if transition and r['minute'] == transition['minute']:
            marker = '  ← ⚠ TRANSITION'
        # 每 5 分鐘或關鍵點印出
        if r['minute'] <= 5 or r['minute'] % 10 == 0 or marker:
            print(f'  t+{r["minute"]:>4}min  {r["alpha"]:>8.3f}  {r["cls_name"]:>12}{marker}')

    print(f'  {"─"*50}')
    if transition:
        old = 'Stable' if current_stable else 'Not-Stable'
        new = transition['cls_name']
        print(f'  ⚠ 穩定度轉換預警: 約 {transition["minute"]} 分鐘後 {old} → {new}')
        print(f'    (predicted α = {transition["alpha"]:.3f})')
    else:
        state = 'Stable' if current_stable else 'Not-Stable'
        print(f'  ✓ 未來 {FORECAST_MAX} 分鐘內穩定度維持 {state}')

# ============================================================
# Parameter Guide (v2)
# ============================================================
# ## Horizon-as-Feature
# 將預報距離 h（分鐘）作為模型輸入特徵之一。
# 訓練時使用 TRAIN_HORIZONS 取樣，推論時可查詢 1~FORECAST_MAX 的任意值。
#
# ## 時間序列特徵
# - 多尺度滾動統計 (5/15/30/60/120 min)
# - 差分 (1st & 2nd order)
# - EWMA (span=10/30/60)
# - FFT 頻域 (主週期 & 低頻能量比)
# - 趨勢斜率 (30/60 min)
#
# ## Expanding Window CV
# 嚴格時序切分 + embargo (=FORECAST_MAX) 防止 target 洩漏。
#
# ## Alpha 物理意義
# WS(z) = WS_ref × (z/z_ref)^α
# α < 0.20 → Not-Stable (對流混合)
# α ≥ 0.20 → Stable (層結穩定)
