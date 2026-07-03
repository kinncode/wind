#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BSMI Virtual Met Mast (虛擬測風塔 / 感測器備援)
- 目標：利用當下 (time t) 的低空風速 (38m, 69m) 與氣象變數 (溫度, 濕度, 氣壓)，
  即時推估 100m 輪轂高度的風速。
- 商業價值：當高空感測器發生故障、結冰或維修時，能利用低空資料精準補值，
  確保風場控制系統與發電量評估不中斷。
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 1. 環境設定
# =====================================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = os.path.join(SCRIPT_DIR, 'BSMI_wind_1min_parquet')
MODEL_DIR   = os.path.join(SCRIPT_DIR, 'models_virtual_mast')
os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================================
# 2. 載入資料
# =====================================================================
print("Loading parquet files...")
files = sorted(glob.glob(os.path.join(PARQUET_DIR, '*.parquet')))
if len(files) == 0:
    raise FileNotFoundError(f"No parquet files found in {PARQUET_DIR}")

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=False).sort_index()
print(f"Total rows: {len(df):,}, Date range: {df.index.min()} ~ {df.index.max()}")

# =====================================================================
# 3. 定義標籤 (Target) 與特徵 (Features)
# =====================================================================
print("Engineering features...")

# 目標：100m 平均風速
df['WS_100_mean'] = (df['WS_100E'] + df['WS_100W']) / 2

# 輸入特徵（禁止使用 100m 的變數，只用低層感測器）
# 基礎風速與氣象變數
base_features = ['WS_69W', 'WS_38W', 'AT_95', 'RH_95', 'BP_93']

# 風向拆分為 sin/cos 避免 360 度與 0 度不連續的問題 (已在資料集中)


# 衍生氣象/動量特徵 (皆為當下或近期的滾動統計)
df['shear_69_38'] = df['WS_69W'] - df['WS_38W']
df['shear_69_38_ratio'] = df['WS_69W'] / (df['WS_38W'] + 0.01)

# 時間特徵 (捕捉日夜/季節變化，這會影響大氣穩定度與風切)
minute_of_day = df.index.hour * 60 + df.index.minute
df['hour_sin'] = np.sin(2 * math.pi * df.index.hour / 24)
df['hour_cos'] = np.cos(2 * math.pi * df.index.hour / 24)
df['month_sin'] = np.sin(2 * math.pi * df.index.month / 12)
df['month_cos'] = np.cos(2 * math.pi * df.index.month / 12)

# 動態特徵：過去10分鐘內的陣風與紊流代理 (不涉及未來)
df['WS_69W_std_10m'] = df['WS_69W'].rolling(10, min_periods=5).std()
df['WS_38W_std_10m'] = df['WS_38W'].rolling(10, min_periods=5).std()
df['AT_95_diff_10m'] = df['AT_95'].diff(10) # 近10分鐘溫度變化

feature_cols = base_features + ['WD_35_sin', 'WD_35_cos', 
                                'shear_69_38', 'shear_69_38_ratio',
                                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                                'WS_69W_std_10m', 'WS_38W_std_10m', 'AT_95_diff_10m']

# 移除缺失值
target_col = 'WS_100_mean'
df_clean = df.dropna(subset=feature_cols + [target_col])
print(f"Data ready for training: {len(df_clean):,} rows")

X = df_clean[feature_cols].values
y = df_clean[target_col].values
times = df_clean.index

# =====================================================================
# 4. 模型訓練與評估 (K-Fold CV)
# =====================================================================
print("\nTraining LightGBM model for Virtual Met Mast...")
lgbm_params = {
    'n_estimators': 1000,
    'learning_rate': 0.5,
    'max_depth': 12,
    'num_leaves': 128,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
models = []

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X[trn_idx], y[trn_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    
    model = LGBMRegressor(**lgbm_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[])
    
    val_preds = model.predict(X_va)
    oof_preds[val_idx] = val_preds
    models.append(model)
    
    rmse = np.sqrt(mean_squared_error(y_va, val_preds))
    r2 = r2_score(y_va, val_preds)
    print(f"Fold {fold+1} | RMSE: {rmse:.4f}, R2: {r2:.4f}")

total_rmse = np.sqrt(mean_squared_error(y, oof_preds))
total_mae = mean_absolute_error(y, oof_preds)
total_r2 = r2_score(y, oof_preds)
print(f"\nOverall CV Performance:")
print(f"RMSE: {total_rmse:.4f} m/s")
print(f"MAE:  {total_mae:.4f} m/s")
print(f"R2:   {total_r2:.4f}")

# 使用所有資料訓練最終模型並儲存
final_model = LGBMRegressor(**lgbm_params)
final_model.fit(X, y)
joblib.dump(final_model, os.path.join(MODEL_DIR, 'Virtual_Met_Mast_LGBM.pkl'))
print("Saved final model.")

# =====================================================================
# 5. 視覺化：散佈圖 (Scatter Plot)
# =====================================================================
print("\nGenerating scatter plot...")
plt.figure(figsize=(8, 8))
# 為避免點太多，隨機抽樣 5 萬筆畫圖
plot_idx = np.random.choice(len(y), size=min(50000, len(y)), replace=False)
plt.scatter(y[plot_idx], oof_preds[plot_idx], alpha=0.1, color='#2E86C1', s=2)
plt.plot([0, 30], [0, 30], 'r--', lw=2, label='Ideal')
plt.xlabel('Actual 100m Wind Speed (m/s)', fontsize=12)
plt.ylabel('Virtual (Predicted) 100m Wind Speed (m/s)', fontsize=12)
plt.title(f'Virtual Met Mast - Validation Scatter Plot\n(R² = {total_r2:.3f}, RMSE = {total_rmse:.3f} m/s)', fontsize=14)
plt.xlim(0, max(y.max(), oof_preds.max())+2)
plt.ylim(0, max(y.max(), oof_preds.max())+2)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'virtual_mast_scatter.png'), dpi=150)
plt.close()

# =====================================================================
# 6. 視覺化：模擬 100m 感測器遺失的情境 (Time-series Trajectory)
# =====================================================================
print("Simulating sensor failure scenario...")
# 找一段連續的 24 小時區間
start_idx = len(df_clean) // 2
end_idx = start_idx + 1440 # 1440 mins = 24 hours
sim_times = times[start_idx:end_idx]
actual_traj = y[start_idx:end_idx]
pred_traj = oof_preds[start_idx:end_idx]

plt.figure(figsize=(15, 5))
plt.plot(sim_times, actual_traj, label='Actual 100m Sensor (Failed in Scenario)', color='gray', alpha=0.6, lw=2)
plt.plot(sim_times, pred_traj, label='Virtual Mast Backup (Imputed)', color='#E74C3C', alpha=0.8, lw=1.5, ls='--')
plt.title('Sensor Imputation Simulation (24-Hour Period)', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Wind Speed (m/s)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'sensor_failure_simulation.png'), dpi=150)
plt.close()

# =====================================================================
# 7. 特徵重要性
# =====================================================================
plt.figure(figsize=(10, 6))
importances = final_model.feature_importances_
idx = np.argsort(importances)
plt.barh(range(len(idx)), importances[idx], color='steelblue')
plt.yticks(range(len(idx)), [feature_cols[i] for i in idx])
plt.xlabel('LightGBM Feature Importance (Split)')
plt.title('Virtual Met Mast Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'), dpi=150)
plt.close()

print("\nDone! Results and models saved to:", MODEL_DIR)
