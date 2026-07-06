"""
============================================================
  低高度風速預測高高度風速 — 隨機森林迴歸完整實作
  Low-height → 100m wind speed prediction (Random Forest)
  
  資料: BSMI 測風塔 1-min 資料 (parquet)
  目標: 用 38m / 69m 風速 + 氣象特徵 → 預測 100m 東風速
  
  執行方式:
    pip install pandas pyarrow numpy scikit-learn matplotlib seaborn
    python wind_rf_regression.py
============================================================
"""

import os
import sys
import glob

# Windows 終端預設 cp950 無法顯示 Unicode 符號，強制 UTF-8
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    mean_absolute_percentage_error
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# 本地路徑設定（自動偵測腳本所在目錄）
# ─────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, 'BSMI_wind_1min_parquet')
OUTPUT_DIR  = SCRIPT_DIR  # 輸出檔案存放目錄

# ─────────────────────────────────────────────────────────
# 0. 設定中文字型（自動偵測系統可用字型）
# ─────────────────────────────────────────────────────────
def setup_chinese_font():
    """自動偵測並設定中文字型"""
    import matplotlib.font_manager as fm
    
    # 常見中文字型優先順序
    candidates = [
        "Microsoft JhengHei",   # Windows 正黑體
        "Microsoft YaHei",      # Windows 雅黑
        "PingFang TC",          # macOS
        "Heiti TC",             # macOS
        "Noto Sans CJK TC",    # Linux
        "Noto Sans TC",        # Linux
        "WenQuanYi Micro Hei", # Linux
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
    print(f"  系統可用字型前 20 個: {sorted(available)[:20]}")
    plt.rcParams["axes.unicode_minus"] = False
    return None

FONT = setup_chinese_font()

# ─────────────────────────────────────────────────────────
# 1. 載入資料
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 | 載入資料")
print("=" * 60)

# 自動掃描 DATA_DIR 下所有 parquet 檔案
DATA_FILES = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
if len(DATA_FILES) == 0:
    raise FileNotFoundError(f"在 {DATA_DIR} 中找不到任何 parquet 檔案")

print(f"  資料目錄: {DATA_DIR}")
print(f"  找到 {len(DATA_FILES)} 個 parquet 檔案")

dfs = []
for f in DATA_FILES:
    dfs.append(pd.read_parquet(f))
    print(f"  ✓ {os.path.basename(f)}: {dfs[-1].shape}")

df = pd.concat(dfs).sort_index()
print(f"\n合併後: {df.shape[0]:,} 筆, {df.shape[1]} 欄")
print(f"期間: {df.index.min()} ~ {df.index.max()}")
print(f"欄位: {list(df.columns)}")

TARGET = "WS_100E"  # 預測目標: 100m 東側風速

# ─────────────────────────────────────────────────────────
# 2. 特徵工程
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 | 特徵工程")
print("=" * 60)

feat = pd.DataFrame(index=df.index)

# ── 2a. 核心: 低高度風速 ──
feat["WS_38W"]  = df["WS_38W"]          # 38m 風速
feat["WS_69W"]  = df["WS_69W"]          # 69m 風速
feat["WS_100W"] = df["WS_100W"]         # 100m 西(同高度對照)

# ── 2b. 亂流強度 TI = σ / μ（從原始 1-min 資料用 10-min 滾動窗口計算） ──
_ws38_std = df["WS_38W"].rolling(10, min_periods=3).std()
_ws69_std = df["WS_69W"].rolling(10, min_periods=3).std()
feat["TI_38W"] = _ws38_std / df["WS_38W"].clip(lower=0.5)
feat["TI_69W"] = _ws69_std / df["WS_69W"].clip(lower=0.5)

# ── 2c. 風切指數 (wind shear exponent) ──
# 冪次律: WS(h) = WS(h_ref) × (h / h_ref)^α
# → α = ln(WS_69 / WS_38) / ln(69 / 38)
with np.errstate(divide="ignore", invalid="ignore"):
    feat["shear_38_69"] = (
        np.log(df["WS_69W"].clip(0.3) / df["WS_38W"].clip(0.3))
        / np.log(69 / 38)
    )
feat["shear_38_69"] = feat["shear_38_69"].clip(-1, 3)

# ── 2d. 風向 sin / cos（已在原始資料中） ──
feat["WD_97_sin"] = df["WD_97_sin"]
feat["WD_97_cos"] = df["WD_97_cos"]
feat["WD_35_sin"] = df["WD_35_sin"]
feat["WD_35_cos"] = df["WD_35_cos"]

# ── 2e. 風向穩定度 r（從 sin/cos 用 10-min 滾動窗口計算結果向量長度） ──
_w = 10
feat["WD_97_r"] = np.sqrt(
    df["WD_97_sin"].rolling(_w, min_periods=3).mean()**2 +
    df["WD_97_cos"].rolling(_w, min_periods=3).mean()**2
)
feat["WD_35_r"] = np.sqrt(
    df["WD_35_sin"].rolling(_w, min_periods=3).mean()**2 +
    df["WD_35_cos"].rolling(_w, min_periods=3).mean()**2
)

# ── 2f. 氣象共變量 ──
feat["AT_95"] = df["AT_95"]    # 氣溫
feat["RH_95"] = df["RH_95"]    # 濕度
feat["BP_93"] = df["BP_93"]    # 氣壓

# ── 2g. 日週期 sin / cos ──
feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

# ── 2h. 滯後特徵（lag）──
for k in [1, 5, 10, 30]:
    feat[f"WS_38W_lag{k}"] = df["WS_38W"].shift(k)
    feat[f"WS_69W_lag{k}"] = df["WS_69W"].shift(k)

# ── 2i. 滑動窗口統計 ──
for w in [10, 30]:
    # 先 shift(1) 防洩漏!
    base38 = df["WS_38W"].shift(1)
    base69 = df["WS_69W"].shift(1)
    feat[f"WS_38W_rmean{w}"] = base38.rolling(w).mean()
    feat[f"WS_69W_rmean{w}"] = base69.rolling(w).mean()
    feat[f"WS_38W_rstd{w}"]  = base38.rolling(w).std()

# ── 2j. 高度比 ──
feat["ratio_69_38"] = df["WS_69W"] / df["WS_38W"].clip(lower=0.3)

# 清除 NaN（lag / rolling 前段不完整）
y = df[TARGET]
feat = feat.dropna()
y = y.loc[feat.index]

print(f"特徵數量: {feat.shape[1]} 個")
print(f"有效樣本: {feat.shape[0]:,} 筆")
print(f"\n特徵列表:")
for i, col in enumerate(feat.columns, 1):
    print(f"  {i:2d}. {col}")

# ─────────────────────────────────────────────────────────
# 3. 時間序列切分（照時間，絕對不 shuffle！）
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 | 時間序列切分 (75% 訓練 / 25% 測試)")
print("=" * 60)

split_idx = int(len(feat) * 0.75)
X_train, X_test = feat.iloc[:split_idx], feat.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"訓練集: {len(X_train):,} 筆  ({X_train.index.min().date()} ~ {X_train.index.max().date()})")
print(f"測試集: {len(X_test):,} 筆  ({X_test.index.min().date()} ~ {X_test.index.max().date()})")

# ─────────────────────────────────────────────────────────
# 4. 訓練隨機森林
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 | 訓練 Random Forest")
print("=" * 60)

rf = RandomForestRegressor(
    n_estimators=300,       # 300 棵樹
    max_depth=12,           # 限制深度防過擬合
    min_samples_leaf=5,     # 葉節點最少 5 個樣本
    max_features="sqrt",    # 每棵樹隨機選 √p 個特徵
    n_jobs=-1,              # 用所有 CPU 核心
    random_state=42,
    verbose=0,
)

print("訓練中...")
rf.fit(X_train, y_train)
print("✓ 訓練完成")

# 預測
y_pred_train = rf.predict(X_train)
y_pred_test  = rf.predict(X_test)

# ─────────────────────────────────────────────────────────
# 5. 評估指標
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 | 模型評估")
print("=" * 60)

def eval_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"\n  [{label}]")
    print(f"  MAE  = {mae:.4f} m/s")
    print(f"  RMSE = {rmse:.4f} m/s")
    print(f"  R²   = {r2:.6f}")
    print(f"  MAPE = {mape:.2f}%")
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE%": mape}

m_train = eval_metrics(y_train, y_pred_train, "訓練集")
m_test  = eval_metrics(y_test,  y_pred_test,  "測試集")

# Naive baseline: Power Law α=1/7
y_naive = X_test["WS_38W"] * (100 / 38) ** (1 / 7)
m_naive = eval_metrics(y_test, y_naive, "Naive Baseline (Power Law α=1/7)")

print(f"\n  → 隨機森林 vs Naive: MAE 降低 {(1 - m_test['MAE']/m_naive['MAE'])*100:.1f}%")

# ── 各風速區間表現 ──
print("\n  各風速區間表現:")
print(f"  {'區間':>12s} | {'筆數':>6s} | {'MAE':>8s} | {'RMSE':>8s}")
print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")
for lo, hi, label in [(0,3,"微風 0-3"), (3,7,"輕風 3-7"), (7,12,"中風 7-12"), (12,20,"強風 12-20"), (20,30,"極端 20+")]:
    mask = (y_test >= lo) & (y_test < hi)
    if mask.sum() > 0:
        mae_b  = mean_absolute_error(y_test[mask], y_pred_test[mask])
        rmse_b = np.sqrt(mean_squared_error(y_test[mask], y_pred_test[mask]))
        print(f"  {label:>12s} | {mask.sum():>6d} | {mae_b:>8.4f} | {rmse_b:>8.4f}")

# ─────────────────────────────────────────────────────────
# 6. Walk-Forward 交叉驗證
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 | Walk-Forward 交叉驗證 (5-fold)")
print("=" * 60)

tscv = TimeSeriesSplit(n_splits=5, test_size=3000)
cv_scores = []

for fold, (tr_idx, te_idx) in enumerate(tscv.split(feat), 1):
    rf_cv = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=42
    )
    rf_cv.fit(feat.iloc[tr_idx], y.iloc[tr_idx])
    p = rf_cv.predict(feat.iloc[te_idx])
    mae_cv = mean_absolute_error(y.iloc[te_idx], p)
    cv_scores.append(mae_cv)
    print(f"  Fold {fold}: train={len(tr_idx):,}, test={len(te_idx):,}, MAE={mae_cv:.4f}")

print(f"\n  CV 平均 MAE = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# ─────────────────────────────────────────────────────────
# 7. 特徵重要性
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 | 特徵重要性")
print("=" * 60)

# 7a. 內建 (Gini impurity)
imp_gini = pd.Series(rf.feature_importances_, index=feat.columns).sort_values(ascending=False)
print("\n  Gini importance Top 15:")
for i, (name, val) in enumerate(imp_gini.head(15).items(), 1):
    bar = "█" * int(val / imp_gini.max() * 30)
    print(f"  {i:2d}. {name:22s} {val:.4f}  {bar}")

# ─────────────────────────────────────────────────────────
# 8. 圖表（6 張子圖）
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 | 繪製分析圖表")
print("=" * 60)

# 配色
C_MAIN   = "#0F7B6C"   # 主色: 深青
C_ACCENT = "#E07A3A"   # 強調: 琥珀橘
C_GRAY   = "#6B7280"   # 灰
C_LIGHT  = "#D1FAE5"   # 淡青底
C_BG     = "#FAFBFC"   # 背景

fig, axes = plt.subplots(3, 2, figsize=(14, 16), facecolor=C_BG)
fig.suptitle(
    "低高度風速 → 100m 風速  |  隨機森林迴歸分析",
    fontsize=17, fontweight="bold", y=0.98, color="#1F2937"
)

# ── 8a. 實際 vs 預測 散布圖 ──
ax = axes[0, 0]
ax.set_facecolor(C_BG)
ax.scatter(y_test.values[::3], y_pred_test[::3], s=3, alpha=0.25, c=C_MAIN, edgecolors="none")
lims = [0, max(y_test.max(), y_pred_test.max()) * 1.05]
ax.plot(lims, lims, "--", color=C_ACCENT, linewidth=1.5, label="y = x", alpha=0.8)
ax.set_xlabel("實際 100m 風速 (m/s)", fontsize=11)
ax.set_ylabel("RF 預測 (m/s)", fontsize=11)
ax.set_title(f"實際 vs 預測  (R² = {m_test['R²']:.4f})", fontsize=13, fontweight="bold")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_aspect("equal")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15)

# ── 8b. 殘差直方圖 ──
ax = axes[0, 1]
ax.set_facecolor(C_BG)
residuals = y_test.values - y_pred_test
ax.hist(residuals, bins=100, color=C_MAIN, alpha=0.7, edgecolor="white", linewidth=0.3, density=True)
ax.axvline(0, color=C_ACCENT, linewidth=1.5, linestyle="--")
mu, sigma = residuals.mean(), residuals.std()
ax.set_xlabel("殘差 = 實際 − 預測 (m/s)", fontsize=11)
ax.set_ylabel("密度", fontsize=11)
ax.set_title(f"殘差分佈  (μ={mu:.4f}, σ={sigma:.4f})", fontsize=13, fontweight="bold")
ax.text(0.97, 0.95, f"99% 殘差落在\n[{np.percentile(residuals,0.5):.2f}, {np.percentile(residuals,99.5):.2f}] m/s",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color=C_GRAY,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#E5E7EB"))
ax.grid(True, alpha=0.15)

# ── 8c. 時間序列追蹤 ──
ax = axes[1, 0]
ax.set_facecolor(C_BG)
n_show = 1440  # 最後 24 小時
t = y_test.index[-n_show:]
ax.plot(t, y_test.values[-n_show:], linewidth=0.7, color=C_GRAY, alpha=0.8, label="實際 100m")
ax.plot(t, y_pred_test[-n_show:], linewidth=0.7, color=C_ACCENT, alpha=0.9, label="RF 預測")
ax.fill_between(t, y_test.values[-n_show:], y_pred_test[-n_show:], alpha=0.12, color=C_ACCENT)
ax.set_ylabel("風速 (m/s)", fontsize=11)
ax.set_title("測試集最後 24 小時追蹤", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.15)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# ── 8d. 各風速區間 MAE ──
ax = axes[1, 1]
ax.set_facecolor(C_BG)
bin_labels, bin_maes, bin_counts = [], [], []
for lo, hi, label in [(0,3,"0-3\n微風"), (3,7,"3-7\n輕風"), (7,12,"7-12\n中風"), (12,20,"12-20\n強風"), (20,30,"20+\n極端")]:
    mask = (y_test >= lo) & (y_test < hi)
    if mask.sum() > 0:
        bin_labels.append(label)
        bin_maes.append(mean_absolute_error(y_test[mask], y_pred_test[mask]))
        bin_counts.append(mask.sum())

colors_bar = [C_MAIN, C_MAIN, C_ACCENT, "#DC2626", "#7C3AED"][:len(bin_labels)]
bars = ax.bar(bin_labels, bin_maes, color=colors_bar, width=0.55, edgecolor="white", linewidth=0.5)
for bar, mae_val, cnt in zip(bars, bin_maes, bin_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{mae_val:.3f}\n(n={cnt:,})", ha="center", va="bottom", fontsize=9, color="#374151")
ax.set_ylabel("MAE (m/s)", fontsize=11)
ax.set_title("各風速區間 MAE", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.15, axis="y")

# ── 8e. 特徵重要性 Top 15 ──
ax = axes[2, 0]
ax.set_facecolor(C_BG)
top15 = imp_gini.head(15).iloc[::-1]
bar_colors = [C_MAIN if v > imp_gini.quantile(0.85) else "#93C5FD" for v in top15.values]
ax.barh(range(len(top15)), top15.values, color=bar_colors, height=0.6, edgecolor="white", linewidth=0.3)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index, fontsize=10)
ax.set_xlabel("Gini Importance", fontsize=11)
ax.set_title("特徵重要性 Top 15", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.15, axis="x")

# ── 8f. Power Law vs ML 比較 ──
ax = axes[2, 1]
ax.set_facecolor(C_BG)
compare_labels = ["Power Law\nα=1/7", "Power Law\nα=0.15", "Power Law\nα=0.20", "Random\nForest"]
compare_maes = []
for alpha_val in [1/7, 0.15, 0.20]:
    p_naive = X_test["WS_38W"] * (100/38) ** alpha_val
    compare_maes.append(mean_absolute_error(y_test, p_naive))
compare_maes.append(m_test["MAE"])

colors_cmp = [C_GRAY, C_GRAY, C_GRAY, C_MAIN]
bars = ax.bar(compare_labels, compare_maes, color=colors_cmp, width=0.5, edgecolor="white")
for bar, val in zip(bars, compare_maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=C_MAIN if val < 0.1 else C_GRAY)
ax.set_ylabel("MAE (m/s)", fontsize=11)
ax.set_title("物理公式 vs 機器學習", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.15, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.96])

# 儲存
output_path = Path(OUTPUT_DIR) / "wind_rf_regression_results.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"\n✓ 圖表已儲存: {output_path}")

plt.show()

# ─────────────────────────────────────────────────────────
# 9. 匯出預測結果
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 | 匯出結果")
print("=" * 60)

result_df = pd.DataFrame({
    "timestamp":  y_test.index,
    "actual_100m": y_test.values,
    "predicted_100m": y_pred_test,
    "residual": residuals,
    "WS_38W":  X_test["WS_38W"].values,
    "WS_69W":  X_test["WS_69W"].values,
    "WS_100W": X_test["WS_100W"].values,
})

csv_path = Path(OUTPUT_DIR) / "wind_rf_predictions.csv"
result_df.to_csv(csv_path, index=False)
print(f"✓ 預測結果已匯出: {csv_path}")
print(f"  共 {len(result_df):,} 筆")

# ─────────────────────────────────────────────────────────
# 10. 最終摘要
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("最終摘要")
print("=" * 60)
print(f"""
  模型:     Random Forest (300 trees, max_depth=12)
  特徵數:   {feat.shape[1]} 個
  訓練集:   {len(X_train):,} 筆
  測試集:   {len(X_test):,} 筆

  ┌─────────────────────────────────────────────┐
  │  測試集指標                                  │
  │  MAE  = {m_test['MAE']:.4f} m/s                        │
  │  RMSE = {m_test['RMSE']:.4f} m/s                        │
  │  R²   = {m_test['R²']:.6f}                          │
  │  MAPE = {m_test['MAPE%']:.2f}%                           │
  ├─────────────────────────────────────────────┤
  │  CV MAE = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}                  │
  ├─────────────────────────────────────────────┤
  │  vs Naive (Power Law α=1/7)                 │
  │  Naive MAE  = {m_naive['MAE']:.4f} m/s                  │
  │  改善幅度   = {(1 - m_test['MAE']/m_naive['MAE'])*100:.1f}%                          │
  └─────────────────────────────────────────────┘

  輸出檔案:
    📊 {output_path.name}
    📄 {csv_path.name}
""")

print("Done! ✓")
