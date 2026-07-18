#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BSMI 風場多任務深度學習時序預報 (Multi-Task Deep Forecasting)
Multi-Task Learning for Wind Speed, Alpha, and Stability Transition Anomaly Warning

本程式包含：
1. 自動檢測與安裝必要套件 (torch, pandas, numpy, scikit-learn 等)
2. 載入 BSMI 測風塔 1-min 資料，動態計算風切指數 alpha (大氣穩定度指標)
3. 進行 10-min Resample，分割出 Train/Test 資料集
4. 構建多任務時序滑動視窗 Dataset：
   - 特徵：過去 144 點 (24 小時) 的風速與 alpha 數值
   - 任務 1：未來 12 點 (2 小時) 的風速預測 (Regression)
   - 任務 2：未來 12 點 (2 小時) 的 alpha 數值預測 (Regression)
   - 任務 3：未來 2 小時內是否會發生大氣穩定度狀態轉換 (Classification, 1=突變, 0=正常)
5. 定義多任務共享神經網路 (CNN-based Shared Extractor + 3 Task-specific Heads)
6. 訓練模型，並評估多個任務的預測精度與分類指標 (MAE, RMSE, F1-Score)
7. 繪製預報曲線與預警混淆矩陣
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix, f1_score
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
# 1. 載入資料與動態計算風切穩定度指數 alpha
# =====================================================================
print("\n" + "=" * 60)
print("STEP 1 | 載入資料並計算大氣穩定度指數 alpha")
print("=" * 60)

files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
if len(files) == 0:
    raise FileNotFoundError(f"在 {DATA_DIR} 中找不到任何 parquet 檔案")

print(f"  載入最新月份 parquet 檔案: {os.path.basename(files[-1])}")
df_raw = pd.read_parquet(files[-1])

# 計算 alpha 指數 (冪律指數，以 38m, 69m, 100m 高度風速擬合)
ws100 = (df_raw['WS_100E'] + df_raw['WS_100W']) / 2
ws69 = df_raw['WS_69W']
ws38 = df_raw['WS_38W']

# 用對數高度與對數風速計算 alpha
HEIGHTS = np.array([38.0, 69.0, 100.0])
LN_HEIGHTS = np.log(HEIGHTS)
WS_mat = np.column_stack([ws38.values, ws69.values, ws100.values])
valid = (WS_mat > 0.1).all(axis=1)

LN_WS = np.log(np.clip(WS_mat, 0.1, None))
lnz_mean = LN_HEIGHTS.mean()
lnws_mean = LN_WS.mean(axis=1)

numer = np.zeros(len(df_raw))
denom = 0.0
for i in range(3):
    numer += (LN_HEIGHTS[i] - lnz_mean) * (LN_WS[:, i] - lnws_mean)
    denom += (LN_HEIGHTS[i] - lnz_mean) ** 2

alpha = numer / denom
alpha[~valid] = np.nan

df_raw['alpha'] = alpha
df_raw['WS_100'] = ws100

# Resample 至 10-min
df_10m = df_raw[['WS_100', 'alpha']].resample('10min').mean().dropna()
print(f"  Resample 至 10-min 後資料長度: {df_10m.shape[0]:,} 筆")

# 使用最後 2016 點做為分析區間
eval_data = df_10m.iloc[-2016:].copy()

# =====================================================================
# 2. 構建多任務時序滑動視窗 Dataset
# =====================================================================
seq_len = 144  # 輸入歷史窗口：144 點 (24 小時)
pred_len = 12  # 預測未來視窗：12 點 (2 小時)
STABLE_THRESHOLD = 0.20 # 穩定狀態界限值 (alpha >= 0.20 為 stable，否則為 unstable)

class MultiTaskWindDataset(Dataset):
    def __init__(self, df, seq_len, pred_len):
        self.ws = df['WS_100'].values
        self.alpha = df['alpha'].values
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.ws) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # 1. 輸入特徵: [Seq_Len, 2] (包含風速與 alpha)
        x_ws = self.ws[idx : idx + self.seq_len]
        x_alpha = self.alpha[idx : idx + self.seq_len]
        x = np.column_stack([x_ws, x_alpha])
        
        # 2. 任務 1 目標: 未來風速 [Pred_Len]
        y_ws = self.ws[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        
        # 3. 任務 2 目標: 未來 alpha [Pred_Len]
        y_alpha = self.alpha[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        
        # 4. 任務 3 目標: 未來是否發生穩定度狀態轉變 (Classification: 0 或 1)
        # 當前狀態
        curr_state = 1 if x_alpha[-1] >= STABLE_THRESHOLD else 0
        # 未來狀態軌跡
        future_states = [1 if val >= STABLE_THRESHOLD else 0 for val in y_alpha]
        # 若未來狀態中任何一點與當前狀態不同，則判定為「發生突變」
        anomaly = 1 if any(s != curr_state for s in future_states) else 0
        
        # 轉成 PyTorch tensor
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_ws, dtype=torch.float32),
            torch.tensor(y_alpha, dtype=torch.float32),
            torch.tensor(anomaly, dtype=torch.float32)
        )

# 時序分割 (按時間劃分：前 70% 時間為訓練集，後 30% 為測試集)
split_idx = int(len(eval_data) * 0.7)
split_time = eval_data.index[split_idx]

train_df = eval_data[eval_data.index < split_time]
test_df = eval_data[eval_data.index >= split_time]

print(f"  資料分割界線時間點: {split_time}")
print(f"  訓練集時間區間: {train_df.index[0]} 至 {train_df.index[-1]} ({len(train_df)} 點)")
print(f"  測試集時間區間: {test_df.index[0]} 至 {test_df.index[-1]} ({len(test_df)} 點)")

train_dataset = MultiTaskWindDataset(train_df, seq_len, pred_len)
test_dataset = MultiTaskWindDataset(test_df, seq_len, pred_len)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"  訓練集滑動視窗數: {len(train_dataset)}")
print(f"  測試集滑動視窗數: {len(test_dataset)}")

# =====================================================================
# 3. 定義多任務共享神經網絡模型 (Multi-Task Net)
# =====================================================================
class MultiTaskNet(nn.Module):
    def __init__(self, seq_len, pred_len, in_channels=2):
        super(MultiTaskNet, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. 共享特徵提取層 (Shared Feature Extractor - 1D CNN)
        # 輸入: [Batch, in_channels=2, Seq_Len]
        self.shared_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2), # -> [Batch, 32, Seq_Len/2]
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8) # 壓縮到固定時間步長 -> [Batch, 64, 8]
        )
        
        # 拉平特徵維度 (64 * 8 = 512)
        self.shared_fc = nn.Sequential(
            nn.Linear(64 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 2. 專屬任務預測頭 (Task-specific Heads)
        # 任務 1: 風速預測 (Regression)
        self.head_wind = nn.Linear(128, self.pred_len)
        
        # 任務 2: alpha 指數預測 (Regression)
        self.head_alpha = nn.Linear(128, self.pred_len)
        
        # 任務 3: 大氣穩定度突變預警 (Classification)
        self.head_anomaly = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, in_channels] -> 轉置以進行 1D 卷積
        x = x.permute(0, 2, 1) # [Batch, in_channels, Seq_Len]
        
        # 共享特徵提取
        feat = self.shared_conv(x)
        feat = feat.view(feat.size(0), -1) # 拉平
        shared_embedding = self.shared_fc(feat) # [Batch, 128]
        
        # 多任務解耦輸出
        out_wind = self.head_wind(shared_embedding)     # [Batch, pred_len]
        out_alpha = self.head_alpha(shared_embedding)   # [Batch, pred_len]
        out_anomaly = self.head_anomaly(shared_embedding).squeeze(-1) # [Batch]
        
        return out_wind, out_alpha, out_anomaly

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskNet(seq_len=seq_len, pred_len=pred_len).to(device)

# 損失函數定義
criterion_reg = nn.MSELoss()
criterion_cls = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# =====================================================================
# 4. 模型多任務聯合訓練
# =====================================================================
print("\n" + "=" * 60)
print(f"STEP 3 | 開始多任務聯合訓練 (Device: {device})")
print("=" * 60)

epochs = 20
# 損失權重比例 (風速 MSE: 1.0, Alpha MSE: 20.0 (因alpha值一般在 0~0.5 較小，需放大), 突警 BCE: 5.0)
w_wind = 1.0
w_alpha = 25.0
w_anomaly = 5.0

for epoch in range(epochs):
    model.train()
    epoch_loss_total = 0.0
    epoch_loss_wind = 0.0
    epoch_loss_alpha = 0.0
    epoch_loss_anomaly = 0.0
    
    for batch_x, batch_y_wind, batch_y_alpha, batch_y_anomaly in train_loader:
        batch_x = batch_x.to(device)
        batch_y_wind = batch_y_wind.to(device)
        batch_y_alpha = batch_y_alpha.to(device)
        batch_y_anomaly = batch_y_anomaly.to(device)
        
        optimizer.zero_grad()
        
        # 前向傳播
        pred_wind, pred_alpha, pred_anomaly = model(batch_x)
        
        # 計算各個任務 Loss
        loss_wind = criterion_reg(pred_wind, batch_y_wind)
        loss_alpha = criterion_reg(pred_alpha, batch_y_alpha)
        loss_anomaly = criterion_cls(pred_anomaly, batch_y_anomaly)
        
        # 聯合權重加總
        total_loss = (w_wind * loss_wind) + (w_alpha * loss_alpha) + (w_anomaly * loss_anomaly)
        
        total_loss.backward()
        optimizer.step()
        
        batch_size = batch_x.size(0)
        epoch_loss_total += total_loss.item() * batch_size
        epoch_loss_wind += loss_wind.item() * batch_size
        epoch_loss_alpha += loss_alpha.item() * batch_size
        epoch_loss_anomaly += loss_anomaly.item() * batch_size
        
    n_samples = len(train_loader.dataset)
    print(f"  Epoch [{epoch+1:02d}/{epochs:02d}] | "
          f"Total: {epoch_loss_total/n_samples:.4f} | "
          f"Wind_MSE: {epoch_loss_wind/n_samples:.4f} | "
          f"Alpha_MSE: {epoch_loss_alpha/n_samples:.4f} | "
          f"Cls_BCE: {epoch_loss_anomaly/n_samples:.4f}")

# =====================================================================
# 5. 多任務模型評估與效能解析
# =====================================================================
print("\n" + "=" * 60)
print("STEP 4 | 多任務預估指標評估")
print("=" * 60)

model.eval()
all_trues_wind = []
all_preds_wind = []
all_trues_alpha = []
all_preds_alpha = []
all_trues_anomaly = []
all_preds_anomaly = []

with torch.no_grad():
    for batch_x, batch_y_wind, batch_y_alpha, batch_y_anomaly in test_loader:
        batch_x = batch_x.to(device)
        p_wind, p_alpha, p_anomaly = model(batch_x)
        
        all_preds_wind.append(p_wind.cpu().numpy())
        all_trues_wind.append(batch_y_wind.numpy())
        all_preds_alpha.append(p_alpha.cpu().numpy())
        all_trues_alpha.append(batch_y_alpha.numpy())
        all_preds_anomaly.append(p_anomaly.cpu().numpy())
        all_trues_anomaly.append(batch_y_anomaly.numpy())

preds_wind = np.concatenate(all_preds_wind, axis=0)
trues_wind = np.concatenate(all_trues_wind, axis=0)
preds_alpha = np.concatenate(all_preds_alpha, axis=0)
trues_alpha = np.concatenate(all_trues_alpha, axis=0)
preds_anomaly = np.concatenate(all_preds_anomaly, axis=0)
trues_anomaly = np.concatenate(all_trues_anomaly, axis=0)

# ---- 任務 1: 風速評估 (10m vs. 120m) ----
print("  [任務 1 - 風速預估軌跡效能 (t+10m)]")
mae_w1 = mean_absolute_error(trues_wind[:, 0], preds_wind[:, 0])
rmse_w1 = np.sqrt(mean_squared_error(trues_wind[:, 0], preds_wind[:, 0]))
r2_w1 = r2_score(trues_wind[:, 0], preds_wind[:, 0])
print(f"    MAE: {mae_w1:.4f} m/s | RMSE: {rmse_w1:.4f} m/s | R²: {r2_w1:.4f}")
print("-" * 55)

# ---- 任務 2: Alpha 穩定度評估 (10m vs. 120m) ----
print("  [任務 2 - 大氣穩定度 alpha 預估效能 (t+10m)]")
mae_a1 = mean_absolute_error(trues_alpha[:, 0], preds_alpha[:, 0])
rmse_a1 = np.sqrt(mean_squared_error(trues_alpha[:, 0], preds_alpha[:, 0]))
r2_a1 = r2_score(trues_alpha[:, 0], preds_alpha[:, 0])
print(f"    MAE: {mae_a1:.4f} | RMSE: {rmse_a1:.4f} | R²: {r2_a1:.4f}")
print("-" * 55)

# ---- 任務 3: 大氣穩定度突變分類預警評估 ----
print("  [任務 3 - 未來 2 小時內穩定度狀態突變分類預警]")
# 閥值設定為 0.5 做為二分類界限
binary_preds = (preds_anomaly >= 0.5).astype(int)
print(classification_report(trues_anomaly, binary_preds, target_names=["正常 (無狀態轉換)", "警報 (穩定度突變)"]))

# =====================================================================
# 6. 繪製多任務視覺化對比圖
# =====================================================================
plot_len = 180
t_index = test_df.index[seq_len : seq_len + plot_len]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), facecolor='#FAFBFC')

# 子圖 1: 風速
ax1.plot(t_index, trues_wind[:plot_len, 0], label='實際 100m 風速', color='#4B5563', lw=1.6)
ax1.plot(t_index, preds_wind[:plot_len, 0], label='MTL 預測風速 (t+10m)', color='#0EA5E9', linestyle='--', lw=1.4)
ax1.set_ylabel('風速 (m/s)')
ax1.legend(loc='upper right')
ax1.grid(alpha=0.15)
ax1.set_title('多任務學習 (Multi-Task Learning) 同步預測與預警展示', fontsize=14, fontweight='bold')

# 子圖 2: Alpha 穩定度
ax2.plot(t_index, trues_alpha[:plot_len, 0], label='實際穩定度 alpha', color='#10B981', lw=1.6)
ax2.plot(t_index, preds_alpha[:plot_len, 0], label='MTL 預測 alpha (t+10m)', color='#F97316', linestyle='--', lw=1.4)
ax2.axhline(y=STABLE_THRESHOLD, color='#EF4444', linestyle=':', label='穩定臨界線 (0.2)')
ax2.set_ylabel('風切指數 alpha')
ax2.legend(loc='upper right')
ax2.grid(alpha=0.15)

# 子圖 3: 狀態轉換預警
ax3.fill_between(t_index, 0, trues_anomaly[:plot_len], color='#EF4444', alpha=0.15, label='實際突變區間')
ax3.plot(t_index, preds_anomaly[:plot_len], label='MTL 突變預警機率', color='#EF4444', lw=1.5)
ax3.axhline(y=0.5, color='#8B5CF6', linestyle='--', label='預警閾值 (0.5)')
ax3.set_ylim(-0.05, 1.05)
ax3.set_ylabel('突變機率 / 標籤')
ax3.legend(loc='upper right')
ax3.grid(alpha=0.15)
ax3.set_xlabel('時間')

plt.tight_layout()
output_chart = os.path.join(OUTPUT_DIR, 'multitask_forecast.png')
plt.savefig(output_chart, dpi=150, facecolor='#FAFBFC')
plt.close()

print(f"\n  ✓ 多任務預報與預警對比圖已儲存至: {output_chart}")
print("=" * 60 + "\n")
