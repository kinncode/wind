#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BSMI 風場時頻信號分解與去噪預報 (VMD & Hybrid RF Forecasting)
Signal Decomposition and Hybrid Machine Learning Forecasting

本程式包含：
1. 變分模態分解 (VMD) 實作，將 100m 風速信號分解成 5 個 IMF 分量
2. 繪製並輸出各分量時序圖與其自相關函數 (ACF) 分析，說明高低頻之統計學特性
3. 實作「無資料洩漏」的滾動窗口 VMD-RF 混合預估模型 (VMD-RF Hybrid)
4. 對比基準隨機森林模型 (Single-RF)，量化時頻訊號分解去噪對預測精度的提升
"""

import os
import sys

# Windows 終端預設 cp950 無法顯示 Unicode 符號，強制 UTF-8
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
from statsmodels.tsa.stattools import acf
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# 本地路徑設定
# ─────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'data', 'BSMI_wind_1min_parquet'))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# 設定中文字型
# ─────────────────────────────────────────────────────────
def setup_chinese_font():
    import matplotlib.font_manager as fm
    candidates = [
        "Microsoft JhengHei",   # Windows 正黑體
        "Microsoft YaHei",      # Windows 雅黑
        "PingFang TC",          # macOS
        "Heiti TC",             # macOS
        "Noto Sans CJK TC",    # Linux
        "Noto Sans TC",        # Linux
        "SimHei",              # Windows 黑體
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = [font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"✓ 使用字型: {font}")
            return font
    print("⚠ 未找到中文字型，圖表文字可能顯示為方塊")
    plt.rcParams["axes.unicode_minus"] = False
    return None

FONT = setup_chinese_font()

# =====================================================================
# 1. 載入資料並 Resample
# =====================================================================
print("\n" + "=" * 60)
print("STEP 1 | 載入資料並 Resample")
print("=" * 60)

files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
if len(files) == 0:
    raise FileNotFoundError(f"在 {DATA_DIR} 中找不到任何 parquet 檔案")

# 讀取最後一個月份的資料做示範以維持快速與連續性
print(f"  載入最新月份 parquet 檔案: {os.path.basename(files[-1])}")
df_raw = pd.read_parquet(files[-1])

# 10-min 區間平均
df_10m = df_raw['WS_100E'].resample('10min').mean().dropna()
print(f"  Resample 至 10-min 後資料長度: {df_10m.shape[0]:,} 筆")

# 為了加快 VMD 計算（VMD 迭代運算在非常長的時間序列上較慢），
# 我們選取一個長度為 2,016 筆的連續區間（剛好為 14 天 / 2 星期 的資料）
eval_series = df_10m.iloc[-2016:]
print(f"  選取分析區間: {eval_series.index.min()} ~ {eval_series.index.max()} (共 {len(eval_series)} 筆)")

# =====================================================================
# 2. 實作 VMD (變分模態分解)
# =====================================================================
print("\n" + "=" * 60)
print("STEP 2 | 執行 VMD (變分模態分解)")
print("=" * 60)

# VMD 參數設定
# alpha: 頻帶寬度限制 (值越小，各模態頻帶越寬)
# tau: 噪聲鬆弛因子 (0 代表無鬆弛約束)
# K: 分解的模態數 (這裡選定 5 個模態)
# DC: 是否保留直流分量 (0 代表不強迫)
# init: 初始化中心頻率 (1 代表均勻分布)
# tol: 收斂容許度
alpha = 2000
tau = 0.0
K = 5
DC = 0
init = 1
tol = 1e-7

print(f"  分解變數: WS_100E, 模態數 K = {K}")
u, u_hat, omega = VMD(eval_series.values, alpha, tau, K, DC, init, tol)

# 轉換為 DataFrame 方便處理
imf_names = [f'IMF_{i+1}' for i in range(K)]
df_imf = pd.DataFrame(u.T, columns=imf_names, index=eval_series.index)

# =====================================================================
# 3. 自相關性分析 (ACF) 證明分頻規律
# =====================================================================
print("\n" + "=" * 60)
print("STEP 3 | IMF 分量的自相關分析 (ACF)")
print("=" * 60)

print(f"  {'分量':^8s} | {'Lag-1 ACF':^10s} | {'Lag-6 ACF (1小時)':^18s} | {'訊號規律特性':^15s}")
print("-" * 65)

acf_results = {}
for col in imf_names:
    # 計算 ACF 值
    acf_vals = acf(df_imf[col], nlags=10)
    acf_results[col] = acf_vals
    
    # 決定特性描述
    lag1 = acf_vals[1]
    lag6 = acf_vals[6]
    if lag1 > 0.9:
        desc = "極高規律 (低頻趨勢)"
    elif lag1 > 0.5:
        desc = "中等規律 (中頻波段)"
    else:
        desc = "低規律 / 隨機噪訊 (高頻)"
        
    print(f"  {col:^8s} | {lag1:^10.4f} | {lag6:^18.4f} | {desc:^15s}")

# =====================================================================
# 4. 繪製分解圖表
# =====================================================================
print("\n" + "=" * 60)
print("STEP 4 | 繪製 VMD 分解結果圖")
print("=" * 60)

# 只畫前 3 天 (432 點) 的波形以便看清局部細節
plot_len = 432
fig, axes = plt.subplots(K + 1, 1, figsize=(12, 12), sharex=True, facecolor='#FAFBFC')

# 原始序列
axes[0].plot(eval_series.index[:plot_len], eval_series.values[:plot_len], color='#4B5563', lw=1.5)
axes[0].set_title('原始 100m 風速序列 (Original Signal)', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.15)

# 5 個 IMF 分量
colors_imf = ['#0EA5E9', '#10B981', '#F97316', '#8B5CF6', '#EF4444']
for i in range(K):
    axes[i+1].plot(df_imf.index[:plot_len], df_imf.iloc[:plot_len, i], color=colors_imf[i], lw=1.2)
    axes[i+1].set_title(f'固有模態函數 {imf_names[i]} (Lag-1 ACF: {acf_results[imf_names[i]][1]:.3f})', fontsize=11)
    axes[i+1].grid(alpha=0.15)

axes[-1].set_xlabel('時間', fontsize=11)
plt.suptitle('變分模態分解 (VMD) 訊號多尺度頻率解構 (展示前 3 天細節)', fontsize=15, fontweight='bold', y=0.985)
plt.tight_layout()

vmd_plot_path = os.path.join(OUTPUT_DIR, 'vmd_decomposition.png')
plt.savefig(vmd_plot_path, dpi=150, facecolor='#FAFBFC')
plt.close()
print(f"  ✓ VMD 分解時序圖已儲存至: {vmd_plot_path}")

# =====================================================================
# 5. VMD-RF 混合預估對比 (無數據洩漏的滾動建模)
# =====================================================================
print("\n" + "=" * 60)
print("STEP 5 | VMD-RF 混合預報 vs. 傳統直接預報")
print("=" * 60)

# 設定滾動預估參數
# 歷史窗口大小為 144 (代表 24 小時的歷史數據)
window_size = 144
# 測試點數為 100 點 (因 VMD 計算較為耗時，測試 100 點約 10-15 秒，具代表性)
test_points = 100

actual_list = []
direct_pred_list = []
direct_lgb_pred_list = []
vmd_hybrid_pred_list = []
vmd_lgb_hybrid_pred_list = []

# 用於測試集之特徵構建
# 使用 Lags 1~6 作為預測特徵
lags = 6

print(f"  開始滾動預報模擬 (測試點數: {test_points} 點, 預測未來 1 步 / 10分鐘)...")

for i in range(test_points):
    # 測試點在 eval_series 中的當前歷史終點位置
    t_end = len(eval_series) - test_points + i
    
    # 實際目標值 (t+1 分鐘的風速)
    actual_val = eval_series.values[t_end]
    actual_list.append(actual_val)
    
    # --- 1a. 傳統直接預報 (Single-RF) ---
    # 取歷史最後 144 點，構建 Lags 特徵
    hist_direct = eval_series.values[t_end - window_size : t_end]
    X_dir_train = []
    y_dir_train = []
    for j in range(lags, len(hist_direct)):
        X_dir_train.append(hist_direct[j - lags : j])
        y_dir_train.append(hist_direct[j])
    
    rf_dir = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf_dir.fit(X_dir_train, y_dir_train)
    x_dir_pred = [hist_direct[-lags:]]
    direct_pred_val = rf_dir.predict(x_dir_pred)[0]
    direct_pred_list.append(direct_pred_val)
    
    # --- 1b. 傳統直接預報 (Single-LGBM) ---
    lgb_dir = LGBMRegressor(n_estimators=50, max_depth=6, num_leaves=31, learning_rate=0.1, random_state=42, verbose=-1, n_jobs=-1)
    lgb_dir.fit(X_dir_train, y_dir_train)
    direct_lgb_pred_val = lgb_dir.predict(x_dir_pred)[0]
    direct_lgb_pred_list.append(direct_lgb_pred_val)
    
    # --- 2. VMD 混合預報 (VMD-RF & VMD-LGBM) ---
    # 對歷史最後 144 點執行 VMD 分解
    u_w, _, _ = VMD(hist_direct, alpha, tau, K, DC, init, tol)
    
    # 對 5 個 IMF 分量分別訓練隨機森林與 LightGBM 進行預估
    imf_forecasts_rf = []
    imf_forecasts_lgb = []
    for k in range(K):
        imf_series = u_w[k] # 當前 IMF 時序
        
        # 構建 lags 特徵
        X_imf_train = []
        y_imf_train = []
        for j in range(lags, len(imf_series)):
            X_imf_train.append(imf_series[j - lags : j])
            y_imf_train.append(imf_series[j])
            
        # VMD-RF 分量擬合
        rf_imf = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1)
        rf_imf.fit(X_imf_train, y_imf_train)
        x_imf_pred = [imf_series[-lags:]]
        imf_forecasts_rf.append(rf_imf.predict(x_imf_pred)[0])
        
        # VMD-LGBM 分量擬合
        lgb_imf = LGBMRegressor(n_estimators=30, max_depth=4, num_leaves=15, learning_rate=0.1, random_state=42, verbose=-1, n_jobs=-1)
        lgb_imf.fit(X_imf_train, y_imf_train)
        imf_forecasts_lgb.append(lgb_imf.predict(x_imf_pred)[0])
        
    # 重組
    vmd_hybrid_pred_list.append(sum(imf_forecasts_rf))
    vmd_lgb_hybrid_pred_list.append(sum(imf_forecasts_lgb))

# 計算效能指標
actual_list = np.array(actual_list)
direct_pred_list = np.array(direct_pred_list)
direct_lgb_pred_list = np.array(direct_lgb_pred_list)
vmd_hybrid_pred_list = np.array(vmd_hybrid_pred_list)
vmd_lgb_hybrid_pred_list = np.array(vmd_lgb_hybrid_pred_list)

mae_dir = mean_absolute_error(actual_list, direct_pred_list)
rmse_dir = np.sqrt(mean_squared_error(actual_list, direct_pred_list))
r2_dir = r2_score(actual_list, direct_pred_list)

mae_lgb = mean_absolute_error(actual_list, direct_lgb_pred_list)
rmse_lgb = np.sqrt(mean_squared_error(actual_list, direct_lgb_pred_list))
r2_lgb = r2_score(actual_list, direct_lgb_pred_list)

mae_vmd = mean_absolute_error(actual_list, vmd_hybrid_pred_list)
rmse_vmd = np.sqrt(mean_squared_error(actual_list, vmd_hybrid_pred_list))
r2_vmd = r2_score(actual_list, vmd_hybrid_pred_list)

mae_vmd_lgb = mean_absolute_error(actual_list, vmd_lgb_hybrid_pred_list)
rmse_vmd_lgb = np.sqrt(mean_squared_error(actual_list, vmd_lgb_hybrid_pred_list))
r2_vmd_lgb = r2_score(actual_list, vmd_lgb_hybrid_pred_list)

metrics_df = pd.DataFrame({
    'Model': [
        'Single-RF (傳統直接預估)', 
        'Single-LGBM (直接預估)', 
        'VMD-RF (時頻分解混合預估)', 
        'VMD-LGBM (時頻分解混合預估)'
    ],
    'MAE (m/s)': [mae_dir, mae_lgb, mae_vmd, mae_vmd_lgb],
    'RMSE (m/s)': [rmse_dir, rmse_lgb, rmse_vmd, rmse_vmd_lgb],
    'R2 Score': [r2_dir, r2_lgb, r2_vmd, r2_vmd_lgb]
})

metrics_csv = os.path.join(OUTPUT_DIR, 'decomposition_forecast_metrics.csv')
metrics_df.to_csv(metrics_csv, index=False, encoding='utf-8-sig')
print(f"  ✓ 預估效能指標計算完成，已儲存至: {metrics_csv}")

print("\n📊 預報指標對比摘要：")
print(metrics_df.to_string(index=False))

# =====================================================================
# 6. 繪製預報比對時序軌跡
# =====================================================================
print("\n" + "=" * 60)
print("STEP 6 | 繪製預測軌跡對比圖")
print("=" * 60)

fig, ax = plt.subplots(figsize=(12, 5), facecolor='#FAFBFC')
ax.set_facecolor('#FAFBFC')
time_axis = eval_series.index[-test_points:]

ax.plot(time_axis, actual_list, 'o-', color='#4B5563', label='實際觀測風速 (Actual)', lw=2)
ax.plot(time_axis, direct_pred_list, 's--', color='#EF4444', label='Single-RF 直接預測', lw=1.2, alpha=0.6)
ax.plot(time_axis, direct_lgb_pred_list, 'v--', color='#F97316', label='Single-LGBM 直接預測', lw=1.2, alpha=0.6)
ax.plot(time_axis, vmd_hybrid_pred_list, 'd:', color='#3B82F6', label='VMD-RF 混合預測', lw=1.2, alpha=0.6)
ax.plot(time_axis, vmd_lgb_hybrid_pred_list, 'h:', color='#10B981', label='VMD-LGBM 混合預測 (SOTA)', lw=1.5)

ax.set_ylabel('100m 風速 (m/s)', fontsize=12)
ax.set_xlabel('時間', fontsize=12)
ax.set_title('時頻訊號分解去噪預報對比圖 (最近 16 小時滾動軌跡)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15)
plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

traj_path = os.path.join(OUTPUT_DIR, 'decomposition_predictions.png')
plt.tight_layout()
plt.savefig(traj_path, dpi=150, facecolor='#FAFBFC')
plt.close()
print(f"  ✓ 軌跡比對圖已儲存至: {traj_path}")

print("\n所有時頻訊號分解與預報工作已完成！ ✓")
