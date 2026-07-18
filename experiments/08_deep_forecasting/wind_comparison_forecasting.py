#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BSMI 風場時序預報模型對比訓練 (SOTA Comparison: DLinear, NLinear, PatchLinear, LightGBM)
Deep Learning & Machine Learning Time-Series Forecasting SOTA Comparison

本程式包含：
1. 自動檢測與安裝必要套件 (torch, pandas, numpy, scikit-learn, lightgbm 等)
2. 載入 BSMI 測風塔 1-min 資料並 Resample 至 10-min
3. 構建 PyTorch 時序滑動視窗 Dataset 與 Tabular 資料
4. 定義 DLinear、NLinear、PatchLinear (簡化通道獨立 Patch 模型)
5. 使用 LightGBM (MultiOutputRegressor) 作為新一代機器學習對標基線
6. 訓練四個模型並評估在測試集的多步預估指標 (MAE, RMSE, R2)
7. 繪製預報軌跡對比圖，儲存至 results/deep_models_comparison.png
8. 輸出評估結果至 CSV 檔案 results/deep_models_comparison_metrics.csv
"""

import os
import sys
import subprocess

# Windows 終端預設 cp950 無法顯示 Unicode 符號，強制 UTF-8
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────
# 自動安裝與檢查套件
# ─────────────────────────────────────────────────────────
def install_packages():
    packages = ["torch", "pyarrow", "pandas", "numpy", "matplotlib", "sklearn", "lightgbm", "tabulate"]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"  正在安裝必要套件 {pkg}...")
            if pkg == "torch":
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
            elif pkg == "sklearn":
                subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

install_packages()

import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
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
# 1. 載入資料與劃分訓練集 / 測試集
# =====================================================================
print("\n" + "=" * 60)
print("STEP 1 | 載入資料與時序資料分割")
print("=" * 60)

files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
if len(files) == 0:
    raise FileNotFoundError(f"在 {DATA_DIR} 中找不到任何 parquet 檔案")

print(f"  載入最新月份 parquet 檔案: {os.path.basename(files[-1])}")
df_raw = pd.read_parquet(files[-1])

# 10-min 區間平均
df_10m = df_raw['WS_100E'].resample('10min').mean().dropna()
print(f"  Resample 至 10-min 後資料長度: {df_10m.shape[0]:,} 筆")

# 使用最後 2016 點做為主要分析集（與之前實驗相同，維持公平性）
eval_series = df_10m.iloc[-2016:]

# 時序分割 (按時間劃分：前 70% 時間為訓練集，後 30% 為測試集)
split_idx = int(len(eval_series) * 0.7)
split_time = eval_series.index[split_idx]

train_df = eval_series[eval_series.index < split_time]
test_df = eval_series[eval_series.index >= split_time]

train_data = train_df.values
test_data = test_df.values

print(f"  資料分割界線時間點: {split_time}")
print(f"  訓練集時間區間: {train_df.index[0]} 至 {train_df.index[-1]} ({len(train_data)} 點)")
print(f"  測試集時間區間: {test_df.index[0]} 至 {test_df.index[-1]} ({len(test_data)} 點)")

# 參數設定
seq_len = 144  # 輸入歷史長度：24 小時
pred_len = 12  # 預測未來長度：2 小時

# =====================================================================
# 2. 構建 PyTorch & Tabular 時序滑動視窗 Dataset
# =====================================================================
class WindDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(-1) # [N, 1]
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x, y

train_dataset = WindDataset(train_data, seq_len, pred_len)
test_dataset = WindDataset(test_data, seq_len, pred_len)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Tabular 資料 (給 LightGBM)
def make_tabular_data(series, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(series) - seq_len - pred_len + 1):
        X.append(series[i : i + seq_len])
        Y.append(series[i + seq_len : i + seq_len + pred_len])
    return np.array(X), np.array(Y)

X_train_tab, Y_train_tab = make_tabular_data(train_data, seq_len, pred_len)
X_test_tab, Y_test_tab = make_tabular_data(test_data, seq_len, pred_len)

# =====================================================================
# 3. 定義深度學習模型
# =====================================================================

# --- DLinear 輔助分解模組 ---
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

# --- 1. DLinear 模型 ---
class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, channels=1, decomp_kernel=25):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decompsition = SeriesDecomp(kernel_size=decomp_kernel)
        self.channels = channels

        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        
        x_out = seasonal_output + trend_output
        return x_out.permute(0, 2, 1)

# --- 2. NLinear 模型 ---
class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len, channels=1):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        seq_last = x[:, -1:, :] # [Batch, 1, Channels]
        x = x - seq_last        # 去中心化 (消去分佈偏移)
        x = x.permute(0, 2, 1)  # [Batch, Channels, Seq_Len]
        x = self.Linear(x)      # [Batch, Channels, Pred_Len]
        x = x.permute(0, 2, 1)  # [Batch, Pred_Len, Channels]
        x = x + seq_last        # 加回偏移
        return x

# --- 3. PatchLinear 模型 ---
class PatchLinear(nn.Module):
    def __init__(self, seq_len, pred_len, channels=1, patch_len=16, stride=8):
        super(PatchLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.patch_len = patch_len
        self.stride = stride
        
        # 計算 patch 數量
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        # 線性映射層：將所有 Patch 的展平表示映射至預測步長
        self.linear = nn.Linear(self.num_patches * self.patch_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        batch_size = x.size(0)
        x = x.permute(0, 2, 1) # [Batch, Channels, Seq_Len]
        
        # 時序分塊：unfold 得到 [Batch, Channels, Num_Patches, Patch_Len]
        x_patched = x.unfold(2, self.patch_len, self.stride)
        
        # 展平所有 Patches
        x_flat = x_patched.reshape(batch_size, self.channels, -1)
        
        # 線性輸出
        out = self.linear(x_flat)
        return out.permute(0, 2, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================================================
# 4. PyTorch 模型訓練與評估通用函數
# =====================================================================
def train_pytorch_model(model_class, model_name, epochs=150, lr=0.0005):
    print(f"\n▸ 訓練深度學習模型: {model_name}...")
    model = model_class(seq_len=seq_len, pred_len=pred_len, channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        if (epoch + 1) % 30 == 0 or epoch == epochs - 1:
            epoch_loss = train_loss / len(train_loader.dataset)
            print(f"  Epoch [{epoch+1:03d}/{epochs:03d}] | MSE Loss: {epoch_loss:.5f}")
            
    # 評估測試集
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.numpy())
            
    preds = np.concatenate(preds, axis=0) # [N, pred_len, 1]
    trues = np.concatenate(trues, axis=0) # [N, pred_len, 1]
    return preds, trues

# =====================================================================
# 5. 開始多模型訓練
# =====================================================================
print("\n" + "=" * 60)
print("STEP 2 | 開始模型訓練與測試集推理")
print("=" * 60)

results = {}

# 1. 訓練 DLinear
preds_dlinear, trues_y = train_pytorch_model(DLinear, "DLinear", epochs=150, lr=0.0005)
results["DLinear"] = preds_dlinear

# 2. 訓練 NLinear
preds_nlinear, _ = train_pytorch_model(NLinear, "NLinear", epochs=150, lr=0.0005)
results["NLinear"] = preds_nlinear

# 3. 訓練 PatchLinear
preds_patchlinear, _ = train_pytorch_model(PatchLinear, "PatchLinear", epochs=150, lr=0.0005)
results["PatchLinear"] = preds_patchlinear

# 4. 訓練 LightGBM (新一代機器學習對標)
print("\n▸ 訓練機器學習模型: LightGBM (MultiOutput)...")
lgbm_base = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1
)
lgbm_model = MultiOutputRegressor(lgbm_base)
lgbm_model.fit(X_train_tab, Y_train_tab)

# 預測 LightGBM
preds_lgbm = lgbm_model.predict(X_test_tab) # [N, pred_len]
results["LightGBM"] = np.expand_dims(preds_lgbm, axis=-1) # 轉為 [N, pred_len, 1]

# =====================================================================
# 6. 計算各時間步效能指標與輸出 CSV
# =====================================================================
print("\n" + "=" * 60)
print("STEP 3 | 計算並儲存預估指標對照表")
print("=" * 60)

metrics_list = []

# 計算未來 10 分鐘 (t+10m, step_idx=0) 與未來 2 小時 (t+120m, step_idx=11) 的 MAE, RMSE, R2
for model_name, preds_arr in results.items():
    for step_idx, step_name in [(0, "t+10min"), (pred_len - 1, "t+120min")]:
        y_true = trues_y[:, step_idx, 0]
        y_pred = preds_arr[:, step_idx, 0]
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        metrics_list.append({
            "Model": model_name,
            "Horizon": step_name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4)
        })

metrics_df = pd.DataFrame(metrics_list)
csv_path = os.path.join(OUTPUT_DIR, 'deep_models_comparison_metrics.csv')
metrics_df.to_csv(csv_path, index=False)
print(f"✓ 已將指標儲存至: {csv_path}")

# 印出結果
print("\n[預估指標彙整對照表]")
print(metrics_df.to_markdown(index=False))

# =====================================================================
# 7. 繪製預測與實際風速對比軌跡圖
# =====================================================================
print("\n" + "=" * 60)
print("STEP 4 | 繪製測試集軌跡對比圖")
print("=" * 60)

plot_len = 150
# 測試集繪圖的時間點 (與測試集滑動視窗目標對齊)
t_index = test_df.index[seq_len : seq_len + plot_len]

plt.figure(figsize=(14, 7), facecolor='#FAFBFC')

# 真實值
plt.plot(t_index, trues_y[:plot_len, 0, 0], label='實際 100m 風速', color='#374151', lw=2.2, zorder=2)

# 各模型在未來 10 分鐘 (t+1) 的預報
colors = {
    "DLinear": "#0EA5E9",      # 晴空藍
    "NLinear": "#EF4444",      # 烈焰紅
    "PatchLinear": "#10B981",  # 翡翠綠
    "LightGBM": "#8B5CF6"      # 皇家紫
}

for model_name, preds_arr in results.items():
    plt.plot(
        t_index, 
        preds_arr[:plot_len, 0, 0], 
        label=f'{model_name} 預測 (t+10min)', 
        color=colors[model_name], 
        linestyle='--', 
        lw=1.6,
        alpha=0.9
    )

plt.title('SOTA 機器學習與深度學習時序預報軌跡對比 (測試集部分展示)', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('時間', fontsize=12)
plt.ylabel('風速 (m/s)', fontsize=12)
plt.legend(fontsize=11, loc='upper right', framealpha=0.95)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

chart_path = os.path.join(OUTPUT_DIR, 'deep_models_comparison.png')
plt.savefig(chart_path, dpi=200, facecolor='#FAFBFC')
plt.close()

print(f"✓ 預測軌跡對比圖已儲存至: {chart_path}")
print("=" * 60 + "\n")
