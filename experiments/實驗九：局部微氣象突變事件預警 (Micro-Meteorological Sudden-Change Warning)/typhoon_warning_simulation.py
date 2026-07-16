#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BSMI 測風塔 2021 璨樹強烈颱風 (Chanthu) 實戰預警動態時間軸模擬
Warning Timeline Simulation for Typhoon Chanthu (September 2021)

本程式包含：
1. 載入 BSMI 測風塔 2016-2021 數據，並計算風切指數 alpha 與氣象前兆特徵
2. 自動分割出最後一折的訓練集，並訓練預警 LightGBM 分類器
3. 提取 2021 年 9 月 璨樹颱風 (Chanthu) 過境時間段：2021-09-10 12:00:00 ~ 2021-09-13 18:00:00
4. 模擬線上運行模式，以 1 分鐘高頻頻率對颱風侵襲前的狀態進行即時預警推理
5. 繪製包含雙預警視窗 (h=30min / h=60min) 的多子圖時序對照圖，儲存至 results 目錄下
"""

import os
import sys
import numpy as np
import pandas as pd
import glob
import warnings
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")

# Windows 終端預設 cp950 無法顯示 Unicode 符號，強制 UTF-8
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────
# 本地路徑設定
# ─────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "BSMI_wind_1min_parquet"))
RESULT_DIR  = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

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
            print(f"[OK] 使用字型: {font}")
            return font
    print("[WARN] 未找到中文字型，圖表文字可能顯示為方塊")
    plt.rcParams["axes.unicode_minus"] = False
    return None

FONT = setup_chinese_font()

# =====================================================================
# 1. 載入資料與事件目錄 (沿用 wind_event_warning.py 定義)
# =====================================================================
print("\n" + "=" * 60)
print("STEP 1 | 載入資料並計算大氣穩定度與前兆特徵")
print("=" * 60)

files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
if len(files) == 0:
    raise FileNotFoundError(f"在 {PARQUET_DIR} 中找不到任何 .parquet 檔案")

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=False).sort_index()
df = df[~df.index.duplicated(keep="first")]
print(f"  載入成功！總數據筆數: {len(df):,} 行")

# 基礎衍生量
df["WS_hub"] = (df["WS_100E"] + df["WS_100W"]) / 2

def circular_shift_deg(sin_s, cos_s, k):
    theta = np.arctan2(sin_s, cos_s)
    d = theta.diff(k)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(d.abs())

df["WD97_shift_30"] = circular_shift_deg(df["WD_97_sin"], df["WD_97_cos"], 30)
df["WD97_shift_60"] = circular_shift_deg(df["WD_97_sin"], df["WD_97_cos"], 60)

# 露點溫度
_RH = np.clip(df["RH_95"], 1, 100)
_g  = (17.27 * df["AT_95"]) / (237.7 + df["AT_95"]) + np.log(_RH / 100)
df["Td_95"]     = 237.7 * _g / (17.27 - _g)
df["T_Td_diff"] = df["AT_95"] - df["Td_95"]

# 差分
df["BP_diff_60"] = df["BP_93"].diff(60)
df["RH_diff_60"] = df["RH_95"].diff(60)
df["WS_diff_30"] = df["WS_hub"].diff(30)

# 載入現成的 Event Catalog
catalog_path = os.path.join(RESULT_DIR, "event_catalog.csv")
if not os.path.exists(catalog_path):
    raise FileNotFoundError("請先執行 wind_event_warning.py 以產出 event_catalog.csv")
catalog = pd.read_csv(catalog_path)
catalog["start"] = pd.to_datetime(catalog["start"])
catalog["end"] = pd.to_datetime(catalog["end"])

# =====================================================================
# 2. 特徵工程與 Dataset 構建
# =====================================================================
def add_ts_features(df):
    feat = pd.DataFrame(index=df.index)
    feat["BP"] = df["BP_93"]
    for k in [10, 30, 60, 180]:
        feat[f"BP_diff_{k}"] = df["BP_93"].diff(k)
    for w in [30, 60, 180]:
        feat[f"BP_std_{w}"] = df["BP_93"].rolling(w, min_periods=w//2).std()
    feat["BP_ewm_30"] = df["BP_93"].ewm(span=30).mean() - df["BP_93"]
    
    feat["RH"] = df["RH_95"]
    for k in [30, 60, 180]:
        feat[f"RH_diff_{k}"] = df["RH_95"].diff(k)
    feat["RH_std_60"] = df["RH_95"].rolling(60, min_periods=30).std()
    feat["AT"] = df["AT_95"]
    for k in [30, 60, 180]:
        feat[f"AT_diff_{k}"] = df["AT_95"].diff(k)
    feat["T_Td"] = df["T_Td_diff"]
    feat["T_Td_diff_60"] = df["T_Td_diff"].diff(60)
    
    feat["WS_hub"] = df["WS_hub"]
    feat["WS_38W"] = df["WS_38W"]
    for k in [10, 30, 60]:
        feat[f"WS_diff_{k}"] = df["WS_hub"].diff(k)
    for w in [10, 30, 60]:
        feat[f"WS_std_{w}"] = df["WS_hub"].rolling(w, min_periods=w//2).std()
    feat["WS_max_60"] = df["WS_hub"].rolling(60, min_periods=30).max()
    feat["TI_hub_30"] = (df["WS_hub"].rolling(30, min_periods=10).std() / df["WS_hub"].clip(lower=0.5))
    
    feat["alpha_38_100"] = (np.log(df["WS_hub"].clip(0.3) / df["WS_38W"].clip(0.3)) / np.log(100.0 / 38.0))
    feat["alpha_roll_30"] = feat["alpha_38_100"].rolling(30, min_periods=15).mean()
    
    feat["WD_97_sin"] = df["WD_97_sin"]
    feat["WD_97_cos"] = df["WD_97_cos"]
    feat["WD97_shift_30"] = df["WD97_shift_30"]
    feat["WD97_shift_60"] = df["WD97_shift_60"]
    feat["WD_97_r"] = np.sqrt(df["WD_97_sin"].rolling(30, min_periods=10).mean() ** 2 + df["WD_97_cos"].rolling(30, min_periods=10).mean() ** 2)
    
    hr  = df.index.hour + df.index.minute / 60
    mon = df.index.month
    feat["hour_sin"]  = np.sin(2 * np.pi * hr / 24)
    feat["hour_cos"]  = np.cos(2 * np.pi * hr / 24)
    feat["month_sin"] = np.sin(2 * np.pi * mon / 12)
    feat["month_cos"] = np.cos(2 * np.pi * mon / 12)
    return feat

print("  正在提取 42 維時間序列特徵...")
feat = add_ts_features(df)
FEATURE_COLS = list(feat.columns) + ["horizon"]

# =====================================================================
# 3. 快速訓練預警分類器 (最後一折)
# =====================================================================
print("\n" + "=" * 60)
print("STEP 2 | 訓練最後一折 LightGBM 預警分類器")
print("=" * 60)

# 重現滾動 Dataset 建立 (只針對訓練期進行採樣以加速)
# 璨樹颱風發生在 2021-09，因此最後一折的訓練集 (時間約 2016 ~ 2020) 能完美涵蓋
split_idx = int(len(df) * 0.8) # 用前 80% 作為訓練
train_df = df.iloc[:split_idx]
train_feat = feat.iloc[:split_idx]

# 尋找訓練集斷段
def find_segments(index, gap_min=1.5):
    gaps = np.where(np.diff(index.values).astype("timedelta64[s]") > np.timedelta64(int(gap_min * 60), "s"))[0]
    starts = np.concatenate([[0], gaps + 1])
    ends   = np.concatenate([gaps, [len(index) - 1]])
    return list(zip(starts, ends))

def build_exclusion_mask(index, catalog):
    mask = np.zeros(len(index), dtype=bool)
    for s, e in zip(catalog["start"], catalog["end"]):
        e_buf = e + pd.Timedelta(minutes=60)
        mask |= (index >= s) & (index <= e_buf)
    return mask

def build_haf_samples(index, feat_df, catalog_df, stride=10):
    event_starts = catalog_df["start"].values.astype("datetime64[ns]")
    excl = build_exclusion_mask(index, catalog_df)
    feat_ok = ~feat_df.isna().any(axis=1).values
    
    X_list, y_list, h_list = [], [], []
    fvals = feat_df.values
    idx64 = index.values.astype("datetime64[ns]")
    horizons = [30, 60]
    segments = find_segments(index)
    
    for seg_s, seg_e in segments:
        lo = seg_s + 180
        hi = seg_e - 60
        if hi <= lo:
            continue
        for pos in range(lo, hi + 1, stride):
            if excl[pos] or not feat_ok[pos]:
                continue
            t = idx64[pos]
            for h in horizons:
                left  = np.searchsorted(event_starts, t, side="right")
                right = np.searchsorted(event_starts, t + np.timedelta64(h, "m"), side="right")
                label = int(right > left)
                X_list.append(fvals[pos])
                y_list.append(label)
                h_list.append(h)
                
    X = np.column_stack([np.asarray(X_list, dtype=np.float32), np.asarray(h_list, dtype=np.float32)])
    y = np.asarray(y_list, dtype=np.int8)
    return X, y

print("  正在構建訓練集滑動視窗樣本...")
X_train, y_train = build_haf_samples(train_df.index, train_feat, catalog, stride=10)
pos_tr = y_train.sum()
spw = (len(y_train) - pos_tr) / max(pos_tr, 1)

print(f"  訓練樣本數: {len(y_train):,}, 正樣本比率: {y_train.mean()*100:.3f}%")
print("  訓練 LightGBM 預警模型中...")

LGBM_PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=8,
                   num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                   min_child_samples=100, random_state=42, verbose=-1)

clf = LGBMClassifier(**LGBM_PARAMS, objective="binary", scale_pos_weight=spw)
clf.fit(X_train, y_train)
print("  ✓ 模型訓練完成！")

# =====================================================================
# 4. 璨樹颱風 (Chanthu) 時間預警動態模擬
# =====================================================================
print("\n" + "=" * 60)
print("STEP 3 | 璨樹颱風 (Chanthu) 時段預警機率動態推估")
print("=" * 60)

# 璨樹颱風影響台灣最顯著的 3 天區間
t_start = "2021-09-10 12:00:00"
t_end   = "2021-09-13 18:00:00"

df_typhoon = df.loc[t_start : t_end]
feat_typhoon = feat.loc[t_start : t_end]

if len(df_typhoon) == 0:
    raise ValueError(f"在 {t_start} 到 {t_end} 之間找不到觀測數據！")

print(f"  璨樹颱風段落長度: {len(df_typhoon)} 分鐘")

# 模擬線上即時預估 (逐分鐘進行推理)
valid_mask = ~feat_typhoon.isna().any(axis=1)
valid_pos = np.where(valid_mask)[0]
eval_times = df_typhoon.index[valid_pos]

# 提取特徵基底
X_base = feat_typhoon.values[valid_pos].astype(np.float32)

# 分別預估 h=30min 和 h=60min 的警報機率
prob_30 = np.zeros(len(eval_times))
prob_60 = np.zeros(len(eval_times))

# h=30 預測
X_30 = np.column_stack([X_base, np.full(len(X_base), 30, dtype=np.float32)])
prob_30 = clf.predict_proba(X_30)[:, 1]

# h=60 預測
X_60 = np.column_stack([X_base, np.full(len(X_base), 60, dtype=np.float32)])
prob_60 = clf.predict_proba(X_60)[:, 1]

# =====================================================================
# 5. 繪製預警時間軸範例圖 (璨樹颱風)
# =====================================================================
print("\n" + "=" * 60)
print("STEP 4 | 繪製璨樹颱風專屬預警時間軸圖表")
print("=" * 60)

# 找出璨樹颱風在 Event Catalog 中對應的突變事件段
# 璨樹颱風登陸前大氣壓於 9/11 下午開始陡降，9/12 達到最強烈風速與最低氣壓
typhoon_events = catalog[
    (catalog["start"] >= pd.Timestamp(t_start)) & 
    (catalog["start"] <= pd.Timestamp(t_end))
]

fig, (ax_bp, ax_rh, ax_ws, ax_prob) = plt.subplots(4, 1, figsize=(14, 12), sharex=True, facecolor='#FAFBFC')

# 1. 氣壓
ax_bp.plot(df_typhoon.index, df_typhoon["BP_93"], color="#1E293B", lw=1.8, label="93m 大氣壓")
ax_bp.set_ylabel("大氣壓 (hPa)", fontsize=11, fontweight='bold')
ax_bp.grid(True, alpha=0.15)
ax_bp.legend(loc="upper right")

# 2. 相對濕度
ax_rh.plot(df_typhoon.index, df_typhoon["RH_95"], color="#0891B2", lw=1.8, label="95m 相對濕度")
ax_rh.set_ylabel("濕度 (%RH)", fontsize=11, fontweight='bold')
ax_rh.grid(True, alpha=0.15)
ax_rh.legend(loc="lower right")

# 3. 輪轂風速與穩定度 alpha
ax_ws.plot(df_typhoon.index, df_typhoon["WS_hub"], color="#B45309", lw=1.8, label="100m 輪轂風速")
ax_ws_twin = ax_ws.twinx()
ax_ws_twin.plot(df_typhoon.index, feat_typhoon["alpha_38_100"], color="#6D28D9", lw=1.2, linestyle=':', alpha=0.7, label="風切指數 alpha")
ax_ws.set_ylabel("風速 (m/s)", fontsize=11, fontweight='bold')
ax_ws_twin.set_ylabel("風切指數 alpha", color="#6D28D9", fontsize=11)
ax_ws.grid(True, alpha=0.15)
ax_ws.legend(loc="upper left")

# 4. 預警機率 (HaF 雙軌)
ax_prob.plot(eval_times, prob_30, color="#EC4899", lw=1.8, label="提前 30 分鐘預警機率 (h=30)")
ax_prob.plot(eval_times, prob_60, color="#8B5CF6", lw=1.8, linestyle='--', label="提前 60 分鐘預警機率 (h=60)")
ax_prob.axhline(y=0.20, color="#EF4444", linestyle=':', label="操作警報閾值 (0.20)") # 使用先前最後一折的最佳閾值
ax_prob.set_ylabel("突變事件預警機率", fontsize=11, fontweight='bold')
ax_prob.set_ylim(-0.05, 1.05)
ax_prob.grid(True, alpha=0.15)
ax_prob.legend(loc="upper left")

# 標記實際偵測出的微氣象突變事件段 (紅色陰影)
for _, r in typhoon_events.iterrows():
    for ax in [ax_bp, ax_rh, ax_ws, ax_prob]:
        ax.axvspan(r["start"], r["end"], color="#EF4444", alpha=0.15)
    # 在氣壓圖上標記事件起點
    ax_bp.annotate("突變事件開始 (風速爆增/氣壓急降)", xy=(r["start"], df.loc[r["start"], "BP_93"]),
                   xytext=(-150, 20), textcoords='offset points',
                   arrowprops=dict(arrowstyle="->", color='#EF4444'), fontsize=10, fontweight='bold', color='#B91C1C')

# 美化時間軸
import matplotlib.dates as mdates
ax_prob.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
plt.xlabel("時間", fontsize=12, fontweight='bold')

plt.suptitle('2021 年璨樹 (Chanthu) 強烈颱風過境 — 離岸風場微氣象突變事件動態預警時間軸模擬', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_plot = os.path.join(RESULT_DIR, "typhoon_chanthu_warning_timeline.png")
plt.savefig(output_plot, dpi=150, facecolor='#FAFBFC')
plt.close()

print(f"\n[OK] 模擬完成！璨樹颱風預警時間軸圖已儲存至:\n     {output_plot}")
print("=" * 60 + "\n")
