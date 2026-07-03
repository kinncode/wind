#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BSMI Virtual Met Mast - Hyperparameter Tuning (最佳預測參數尋找)
- 目標：使用 Optuna 對 LightGBM 進行超參數最佳化 (Hyperparameter Tuning)。
- 特色：由於原始資料高達 270 萬筆，為了加速尋找參數的過程，
  此腳本預設會進行隨機抽樣 (例如抽取 20% 資料)，並使用 3 折交叉驗證 (3-Fold CV)。
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import warnings
import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.INFO)

# =====================================================================
# 1. 環境設定
# =====================================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = os.path.join(SCRIPT_DIR, 'BSMI_wind_1min_parquet')

# 參數設定
SAMPLE_FRACTION = 0.2  # 抽樣比例 (20%)，如果運算資源足夠可設為 1.0
N_TRIALS = 30          # Optuna 尋找次數
N_SPLITS = 3           # 交叉驗證折數

# =====================================================================
# 2. 載入資料與特徵工程 (與主腳本相同)
# =====================================================================
print("Loading parquet files...")
files = sorted(glob.glob(os.path.join(PARQUET_DIR, '*.parquet')))
if len(files) == 0:
    raise FileNotFoundError(f"No parquet files found in {PARQUET_DIR}")

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=False).sort_index()

print("Engineering features...")
df['WS_100_mean'] = (df['WS_100E'] + df['WS_100W']) / 2
base_features = ['WS_69W', 'WS_38W', 'AT_95', 'RH_95', 'BP_93']
df['shear_69_38'] = df['WS_69W'] - df['WS_38W']
df['shear_69_38_ratio'] = df['WS_69W'] / (df['WS_38W'] + 0.01)

df['hour_sin'] = np.sin(2 * math.pi * df.index.hour / 24)
df['hour_cos'] = np.cos(2 * math.pi * df.index.hour / 24)
df['month_sin'] = np.sin(2 * math.pi * df.index.month / 12)
df['month_cos'] = np.cos(2 * math.pi * df.index.month / 12)

df['WS_69W_std_10m'] = df['WS_69W'].rolling(10, min_periods=5).std()
df['WS_38W_std_10m'] = df['WS_38W'].rolling(10, min_periods=5).std()
df['AT_95_diff_10m'] = df['AT_95'].diff(10)

feature_cols = base_features + ['WD_35_sin', 'WD_35_cos', 
                                'shear_69_38', 'shear_69_38_ratio',
                                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                                'WS_69W_std_10m', 'WS_38W_std_10m', 'AT_95_diff_10m']
target_col = 'WS_100_mean'

df_clean = df.dropna(subset=feature_cols + [target_col])
print(f"Total valid rows: {len(df_clean):,}")

# 為了加速搜尋，進行隨機抽樣
if SAMPLE_FRACTION < 1.0:
    print(f"Sampling {SAMPLE_FRACTION*100}% of data for faster tuning...")
    df_clean = df_clean.sample(frac=SAMPLE_FRACTION, random_state=42)
    print(f"Sampled rows: {len(df_clean):,}")

X = df_clean[feature_cols].values
y = df_clean[target_col].values

# =====================================================================
# 3. 定義 Optuna 目標函數
# =====================================================================
def objective(trial):
    # 定義 LightGBM 搜尋的超參數範圍
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1  # 使用所有 CPU 核心
    }
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_rmse = []
    
    for trn_idx, val_idx in kf.split(X, y):
        X_tr, y_tr = X[trn_idx], y[trn_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        model = LGBMRegressor(**params)
        # 加入 early_stopping 防止過度擬合並加速
        model.fit(
            X_tr, y_tr, 
            eval_set=[(X_va, y_va)], 
            callbacks=[]
        )
        
        preds = model.predict(X_va)
        rmse = np.sqrt(mean_squared_error(y_va, preds))
        fold_rmse.append(rmse)
    
    # 回傳平均 RMSE 作為優化目標 (越小越好)
    return np.mean(fold_rmse)

# =====================================================================
# 4. 執行最佳化
# =====================================================================
if __name__ == "__main__":
    print(f"\nStarting Optuna hyperparameter tuning ({N_TRIALS} trials)...")
    
    # 建立一個 study (目標為最小化 RMSE)
    study = optuna.create_study(direction='minimize', study_name="BSMI_Virtual_Mast")
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("\n" + "="*50)
    print("Tuning Completed!")
    print("="*50)
    print(f"Best RMSE Score: {study.best_value:.4f} m/s")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")
    
    print("\n(請將上述最佳參數複製並替換至 BSMI_Virtual_Met_Mast.py 中的 lgbm_params)")
