#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BSMI 風場傳統時序統計分析與預報 (ARIMA & VAR)
Traditional Time-Series Forecasting (ARIMA & Vector Autoregression)

本程式包含：
1. ADF 平穩性檢定
2. ACF / PACF 自相關性繪圖
3. ARIMA(2, 1, 2) 單變量風速預報 (10 ~ 60 分鐘)
4. VAR(向量自迴歸) 多高度聯立風速預報 (10 ~ 60 分鐘)
5. 預報效能評估 (MAE, RMSE, R2) 與統計圖表輸出
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
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
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
# 1. 載入資料與 Resample
# =====================================================================
print("\n" + "=" * 60)
print("STEP 1 | 載入資料並 Resample 至 10 分鐘")
print("=" * 60)

files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
if len(files) == 0:
    raise FileNotFoundError(f"在 {DATA_DIR} 中找不到任何 parquet 檔案")

# 讀取最後 3 個檔案（約 3 個月）的資料以平衡分析速度與連續性
print(f"  載入最後 3 個 parquet 檔案...")
dfs = []
for f in files[-3:]:
    dfs.append(pd.read_parquet(f))
    print(f"    ✓ {os.path.basename(f)}")

df_raw = pd.concat(dfs).sort_index()
print(f"  合併後資料長度: {df_raw.shape[0]:,} 筆")

# 10-min 區間取平均
df_10m = df_raw[['WS_100E', 'WS_69W', 'WS_38W']].resample('10min').mean().dropna()
print(f"  Resample 至 10-min 後有效資料長度: {df_10m.shape[0]:,} 筆")

# =====================================================================
# 2. ADF 平穩性檢定 (Augmented Dickey-Fuller Test)
# =====================================================================
print("\n" + "=" * 60)
print("STEP 2 | 進行 ADF 平穩性檢定")
print("=" * 60)

target_col = 'WS_100E'
adf_res = adfuller(df_10m[target_col])
print(f"  變數: {target_col}")
print(f"  ADF 統計量 (Test Statistic): {adf_res[0]:.4f}")
print(f"  p-value: {adf_res[1]:.4e}")
print(f"  Lags Used: {adf_res[2]}")
print(f"  Number of Observations: {adf_res[3]}")
print("  臨界值 (Critical Values):")
for key, val in adf_res[4].items():
    print(f"    {key}: {val:.4f}")

if adf_res[1] < 0.05:
    print(f"  ➔ p-value < 0.05，拒絕虛無假設 (Null Hypothesis)，序列為【平穩的 (Stationary)】，不需強迫進行差分。")
else:
    print(f"  ➔ p-value >= 0.05，無法拒絕虛無假設，序列為【非平穩的 (Non-Stationary)】，建議進行一階差分。")

# =====================================================================
# 3. 繪製 ACF / PACF 相關圖
# =====================================================================
print("\n" + "=" * 60)
print("STEP 3 | 繪製 ACF 與 PACF 自相關分析圖")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#FAFBFC')
plot_acf(df_10m[target_col], lags=40, ax=axes[0], color='#0EA5E9', title='自相關函數 (ACF)')
plot_pacf(df_10m[target_col], lags=40, ax=axes[1], color='#F97316', title='偏自相關函數 (PACF)')
axes[0].grid(alpha=0.15)
axes[1].grid(alpha=0.15)

acf_path = os.path.join(OUTPUT_DIR, 'acf_pacf_plots.png')
plt.tight_layout()
plt.savefig(acf_path, dpi=150, facecolor='#FAFBFC')
plt.close()
print(f"  ✓ ACF / PACF 分析圖已儲存至: {acf_path}")

# =====================================================================
# 4. 時序預估建模 (ARIMA & VAR)
# =====================================================================
print("\n" + "=" * 60)
print("STEP 4 | 切分訓練與測試集 (80/20)")
print("=" * 60)

train_len = int(len(df_10m) * 0.8)
train = df_10m.iloc[:train_len]
test = df_10m.iloc[train_len:]

print(f"  訓練集長度: {len(train):,} 筆 ({train.index.min()} ~ {train.index.max()})")
print(f"  測試集長度: {len(test):,} 筆 ({test.index.min()} ~ {test.index.max()})")

# ─────────────────────────────────────────────────────────
# 4a. ARIMA 模型預測
# ─────────────────────────────────────────────────────────
print("\n  訓練 ARIMA(2, 1, 2) 模型中...")
arima_model = ARIMA(train[target_col], order=(2, 1, 2))
arima_results = arima_model.fit()
print("  ✓ ARIMA 模型擬合完成")

# ─────────────────────────────────────────────────────────
# 4b. VAR 模型預測
# ─────────────────────────────────────────────────────────
print("\n  訓練 VAR 模型中...")
var_model = VAR(train[['WS_38W', 'WS_69W', 'WS_100E']])
selected_order = var_model.select_order(maxlags=10)
best_lag = selected_order.aic
print(f"  ➔ 基於 AIC 選擇最佳滯後階數 (Lag Order): {best_lag}")
var_results = var_model.fit(best_lag)
print("  ✓ VAR 模型擬合完成")

# =====================================================================
# 5. 滾動多步預報評估 (1 ~ 6 steps, 10 ~ 60 mins)
# =====================================================================
print("\n" + "=" * 60)
print("STEP 5 | 滾動多步預報評估 (預測未來 10 ~ 60 分鐘)")
print("=" * 60)

# 為加快評估速度，在測試集中抽樣 500 個時間點進行 1~6 步預測評估
eval_points = min(500, len(test) - 6)
eval_indices = np.linspace(0, len(test) - 7, eval_points, dtype=int)

# 儲存預測結果與實際值
# 結構為 [horizon_index][point_index]
actual_matrix = {h: [] for h in range(1, 7)}
arima_pred_matrix = {h: [] for h in range(1, 7)}
var_pred_matrix = {h: [] for h in range(1, 7)}

# 為了在 statsmodels 中不重複擬合參數，我們使用結果對象的 apply 方法來更新時序狀態
# 這比在循環中每次重建/擬合模型快上百倍
print("  開始滾動預報模擬...")
for idx in eval_indices:
    # 當前的時間點位置（在整個 df_10m 中的索引）
    current_pos = train_len + idx
    
    # 實際未來 1~6 步的值
    futures = df_10m[target_col].values[current_pos + 1 : current_pos + 7]
    for h in range(1, 7):
        actual_matrix[h].append(futures[h-1])
        
    # --- ARIMA 預測 ---
    # 建立歷史觀測數據（到當前時間點為止）
    history_arima = df_10m[target_col].iloc[:current_pos + 1]
    # 更新狀態不重擬合參數
    updated_arima = arima_results.apply(history_arima, refit=False)
    # 預報未來 6 步
    arima_forecast = updated_arima.forecast(steps=6)
    for h in range(1, 7):
        arima_pred_matrix[h].append(arima_forecast.iloc[h-1])
        
    # --- VAR 預測 ---
    history_var = df_10m[['WS_38W', 'WS_69W', 'WS_100E']].iloc[:current_pos + 1]
    updated_var = var_results # VAR 預估可直接利用最終擬合參數，傳入歷史最後 best_lag 筆數據預估
    lagged_values = history_var.values[-best_lag:]
    var_forecast = updated_var.forecast(y=lagged_values, steps=6)
    # VAR 變數順序：WS_38W(0), WS_69W(1), WS_100E(2)
    for h in range(1, 7):
        var_pred_matrix[h].append(var_forecast[h-1, 2])

# 計算評估指標
metrics_data = []

for h in range(1, 7):
    act = np.array(actual_matrix[h])
    pred_ari = np.array(arima_pred_matrix[h])
    pred_var = np.array(var_pred_matrix[h])
    
    # ARIMA 指標
    mae_ari = mean_absolute_error(act, pred_ari)
    rmse_ari = np.sqrt(mean_squared_error(act, pred_ari))
    r2_ari = r2_score(act, pred_ari)
    
    # VAR 指標
    mae_var = mean_absolute_error(act, pred_var)
    rmse_var = np.sqrt(mean_squared_error(act, pred_var))
    r2_var = r2_score(act, pred_var)
    
    metrics_data.append({
        'Horizon_min': h * 10,
        'ARIMA_MAE': mae_ari,
        'ARIMA_RMSE': rmse_ari,
        'ARIMA_R2': r2_ari,
        'VAR_MAE': mae_var,
        'VAR_RMSE': rmse_var,
        'VAR_R2': r2_var
    })

metrics_df = pd.DataFrame(metrics_data)
metrics_csv_path = os.path.join(OUTPUT_DIR, 'statistical_forecast_metrics.csv')
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"  ✓ 預報指標評估完成，已儲存至: {metrics_csv_path}")

print("\n📊 預報指標結果摘要：")
print(metrics_df.to_string(index=False))

# =====================================================================
# 6. 視覺化預測軌跡對比
# =====================================================================
print("\n" + "=" * 60)
print("STEP 6 | 繪製預測軌跡圖")
print("=" * 60)

# 選取一段連續的 48 步（8 小時）在測試集中的實際觀測與預測做對比
# 我們用最後一筆評估點作為展示起點
sp = train_len + eval_indices[-1]
ts_indices = df_10m.index[sp + 1 : sp + 7]

pred_t_arima = arima_pred_matrix[1][-24:] # 取後24筆單步預測
actual_t = actual_matrix[1][-24:]
pred_t_var = var_pred_matrix[1][-24:]
time_axis = df_10m.index[train_len + eval_indices[-24:] + 1]

fig, ax = plt.subplots(figsize=(12, 5), facecolor='#FAFBFC')
ax.set_facecolor('#FAFBFC')
ax.plot(time_axis, actual_t, 'o-', color='#4B5563', label='實際觀測 (Actual)', lw=2)
ax.plot(time_axis, pred_t_arima, 's--', color='#0EA5E9', label='ARIMA(2,1,2) 預測', lw=1.5)
ax.plot(time_axis, pred_t_var, 'd:', color='#F97316', label='VAR 聯立預測', lw=1.5)
ax.set_ylabel('100m 風速 (m/s)', fontsize=12)
ax.set_xlabel('時間', fontsize=12)
ax.set_title('ARIMA vs VAR 單步預報對比 (最近 4 小時時序追蹤)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15)
plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

traj_path = os.path.join(OUTPUT_DIR, 'arima_var_predictions.png')
plt.tight_layout()
plt.savefig(traj_path, dpi=150, facecolor='#FAFBFC')
plt.close()
print(f"  ✓ 預測時序對比圖已儲存至: {traj_path}")

print("\n所有時序統計分析與預報工作已完成！ ✓")
