#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BSMI 風場深度學習時序預報 (DLinear SOTA Forecasting)
Deep Learning Time-Series Forecasting via DLinear Model

本程式包含：
1. 自動檢測與動態安裝 torch 等必要依賴套件
2. 載入 BSMI 測風塔 1-min 資料並 Resample 至 10-min
3. 構建 PyTorch 時序滑動視窗 Dataset (輸入過去 144 點 / 24小時，預測未來 12 點 / 2小時)
4. 定義 DLinear 深度學習時序預報模型 (Trend & Seasonal 雙線性分解結構)
5. 執行訓練與測試，並計算 MAE, RMSE, R2 指標
6. 繪製預測與實際風速對比軌跡圖，存至 results 目錄下
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
    packages = ["torch", "pyarrow", "pandas", "numpy", "matplotlib", "scikit-learn"]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"  正在安裝必要套件 {pkg}...")
            # 針對 PyTorch 安裝速度優化
            if pkg == "torch":
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
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
print("STEP 1 | 載入資料並進行時序分割")
print("=" * 60)

files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
if len(files) == 0:
    raise FileNotFoundError(f"在 {DATA_DIR} 中找不到任何 parquet 檔案")

print(f"  載入最新月份 parquet 檔案: {os.path.basename(files[-1])}")
df_raw = pd.read_parquet(files[-1])

# 10-min 區間平均
df_10m = df_raw['WS_100E'].resample('10min').mean().dropna()
print(f"  Resample 至 10-min 後資料長度: {df_10m.shape[0]:,} 筆")

# 使用最後 2016 點做為主要分析集（與 VMD 實驗相同長度，維持比對公平性）
eval_series = df_10m.iloc[-2016:].values

# 時序分割 (70% 訓練, 30% 測試，不打亂以防止資訊洩漏)
split_idx = int(len(eval_series) * 0.7)
train_data = eval_series[:split_idx]
test_data = eval_series[split_idx:]

print(f"  訓練集長度: {len(train_data)} 點 (~9.8 天)")
print(f"  測試集長度: {len(test_data)} 點 (~4.2 天)")

# =====================================================================
# 2. 構建 PyTorch 時序滑動視窗 Dataset
# =====================================================================
seq_len = 144  # 輸入歷史長度：144 點 (24 小時)
pred_len = 12  # 預測未來長度：12 點 (2 小時)

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

# =====================================================================
# 3. 定義 DLinear 深度學習模型
# =====================================================================
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
        # x: [Batch, Seq_Len, Channels]
        seasonal_init, trend_init = self.decompsition(x)
        
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        
        x_out = seasonal_output + trend_output
        return x_out.permute(0, 2, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DLinear(seq_len=seq_len, pred_len=pred_len, channels=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# =====================================================================
# 4. 模型訓練
# =====================================================================
print("\n" + "=" * 60)
print(f"STEP 2 | 開始訓練 DLinear 模型 (Device: {device})")
print("=" * 60)

epochs = 200
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
    
    epoch_loss = train_loss / len(train_loader.dataset)
    print(f"  Epoch [{epoch+1:02d}/{epochs:02d}] | Train MSE Loss: {epoch_loss:.5f}")

# =====================================================================
# 5. 模型評估 (測試集多步預報)
# =====================================================================
print("\n" + "=" * 60)
print("STEP 3 | 在測試集上執行未來 2 小時預測與評估")
print("=" * 60)

model.eval()
preds = []
trues = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        
        preds.append(outputs.cpu().numpy())
        trues.append(batch_y.numpy())

preds = np.concatenate(preds, axis=0) # [Test_Samples, pred_len, 1]
trues = np.concatenate(trues, axis=0) # [Test_Samples, pred_len, 1]

# 針對「第一步」(未來 10 分鐘) 與「最後一步」(未來 2 小時) 進行指標計算
for step_idx, step_name in [(0, "未來 10 分鐘 (t+1)"), (pred_len - 1, "未來 2 小時 (t+12)")]:
    y_true = trues[:, step_idx, 0]
    y_pred = preds[:, step_idx, 0]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"  ▸ {step_name} 預報效能:")
    print(f"    - MAE:  {mae:.4f} m/s")
    print(f"    - RMSE: {rmse:.4f} m/s")
    print(f"    - R²:   {r2:.4f}")
    print("-" * 40)

# =====================================================================
# 6. 繪製預測對比圖表
# =====================================================================
# 我們提取測試集中的連續 200 個時間點，繪製實際值與模型在 t+1 (未來 10m) 預測的對比
plot_len = 200
t_index = df_10m.index[split_idx + seq_len : split_idx + seq_len + plot_len]

plt.figure(figsize=(12, 6), facecolor='#FAFBFC')
plt.plot(t_index, trues[:plot_len, 0, 0], label='實際 100m 風速', color='#4B5563', lw=1.8)
plt.plot(t_index, preds[:plot_len, 0, 0], label='DLinear 預測 (t+10m)', color='#0EA5E9', linestyle='--', lw=1.5)
plt.title('DLinear 深度學習多步時序預報軌跡對比 (測試集部分展示)', fontsize=14, fontweight='bold')
plt.xlabel('時間', fontsize=11)
plt.ylabel('風速 (m/s)', fontsize=11)
plt.legend(fontsize=10, loc='upper right')
plt.grid(alpha=0.15)
plt.tight_layout()

chart_path = os.path.join(OUTPUT_DIR, 'dlinear_forecast.png')
plt.savefig(chart_path, dpi=150, facecolor='#FAFBFC')
plt.close()
print(f"  ✓ 預測對比軌跡圖已儲存至: {chart_path}")

# 將指標儲存到 CSV
results_df = pd.DataFrame({
    'Horizon': [f't+{i*10}min' for i in range(1, pred_len + 1)],
    'MAE': [mean_absolute_error(trues[:, i, 0], preds[:, i, 0]) for i in range(pred_len)],
    'RMSE': [np.sqrt(mean_squared_error(trues[:, i, 0], preds[:, i, 0])) for i in range(pred_len)],
    'R2': [r2_score(trues[:, i, 0], preds[:, i, 0]) for i in range(pred_len)]
})
csv_path = os.path.join(OUTPUT_DIR, 'dlinear_metrics.csv')
results_df.to_csv(csv_path, index=False)
print(f"  ✓ 預報指標資料表已儲存至: {csv_path}")
print("=" * 60 + "\n")
