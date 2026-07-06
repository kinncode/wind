"""
============================================================
  風況分群分析 — 找出典型天氣模式
  Wind Condition Clustering — Identify Typical Weather Patterns

  資料: BSMI 測風塔 1-min 資料 (parquet)
  方法: 10-min 彙總 → StandardScaler → PCA → K-Means
  
  執行方式:
    pip install pandas pyarrow numpy scikit-learn matplotlib seaborn
    python wind_clustering.py
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
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# 本地路徑設定
# ─────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, 'BSMI_wind_1min_parquet')
OUTPUT_DIR  = SCRIPT_DIR

# ─────────────────────────────────────────────────────────
# 0. 設定中文字型（自動偵測系統可用字型）
# ─────────────────────────────────────────────────────────
def setup_chinese_font():
    """自動偵測並設定中文字型"""
    import matplotlib.font_manager as fm
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
    plt.rcParams["axes.unicode_minus"] = False
    return None

FONT = setup_chinese_font()

# ─────────────────────────────────────────────────────────
# 1. 載入資料
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 | 載入資料")
print("=" * 60)

DATA_FILES = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
if len(DATA_FILES) == 0:
    raise FileNotFoundError(f"在 {DATA_DIR} 中找不到任何 parquet 檔案")

print(f"  資料目錄: {DATA_DIR}")
print(f"  找到 {len(DATA_FILES)} 個 parquet 檔案")

dfs = []
for f in DATA_FILES:
    dfs.append(pd.read_parquet(f))
    print(f"  ✓ {os.path.basename(f)}: {dfs[-1].shape}")

df_raw = pd.concat(dfs).sort_index()
print(f"\n合併後: {df_raw.shape[0]:,} 筆, {df_raw.shape[1]} 欄")
print(f"期間: {df_raw.index.min()} ~ {df_raw.index.max()}")

# ─────────────────────────────────────────────────────────
# 2. 10-min 彙總 + 特徵工程
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 | 10-min 彙總 + 分群特徵工程")
print("=" * 60)

# 10-min resample
agg_dict = {
    'WS_100E': ['mean', 'std'],
    'WS_100W': ['mean'],
    'WS_69W':  ['mean', 'std'],
    'WS_38W':  ['mean'],
    'AT_95':   ['mean'],
    'RH_95':   ['mean'],
    'BP_93':   ['mean'],
    'WD_97_sin': ['mean'],
    'WD_97_cos': ['mean'],
    'WD_35_sin': ['mean'],
    'WD_35_cos': ['mean'],
}

df_10 = df_raw.resample('10min').agg(agg_dict)
df_10.columns = ['_'.join(col).strip() for col in df_10.columns.values]
df_10 = df_10.dropna()

print(f"  10-min 彙總後: {df_10.shape[0]:,} 筆, {df_10.shape[1]} 欄")

# 構建分群特徵
feat = pd.DataFrame(index=df_10.index)

# 各高度風速均值
feat["WS_100E"] = df_10["WS_100E_mean"]
feat["WS_69W"]  = df_10["WS_69W_mean"]
feat["WS_38W"]  = df_10["WS_38W_mean"]

# 亂流指標: 10-min 標準差 / 均值
feat["TI_100E"] = df_10["WS_100E_std"] / df_10["WS_100E_mean"].clip(lower=0.5)
feat["TI_69W"]  = df_10["WS_69W_std"]  / df_10["WS_69W_mean"].clip(lower=0.5)

# 風切指數 α = ln(WS_69/WS_38) / ln(69/38)
with np.errstate(divide="ignore", invalid="ignore"):
    feat["shear_38_69"] = (
        np.log(df_10["WS_69W_mean"].clip(0.3) / df_10["WS_38W_mean"].clip(0.3))
        / np.log(69 / 38)
    )
feat["shear_38_69"] = feat["shear_38_69"].clip(-1, 3)

# 風向 (向量平均)
feat["WD_97_sin"] = df_10["WD_97_sin_mean"]
feat["WD_97_cos"] = df_10["WD_97_cos_mean"]

# 氣象共變量
feat["AT_95"] = df_10["AT_95_mean"]
feat["BP_93"] = df_10["BP_93_mean"]

# 日週期
feat["hour_sin"] = np.sin(2 * np.pi * df_10.index.hour / 24)
feat["hour_cos"] = np.cos(2 * np.pi * df_10.index.hour / 24)

# 清除殘餘 NaN
feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

print(f"  分群特徵數: {feat.shape[1]} 個")
print(f"  有效樣本: {feat.shape[0]:,} 筆")
print(f"\n  特徵列表:")
for i, col in enumerate(feat.columns, 1):
    print(f"    {i:2d}. {col}")

# ─────────────────────────────────────────────────────────
# 3. 標準化 + PCA 降維
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 | 標準化 + PCA 降維")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(feat)

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"  原始維度: {X_scaled.shape[1]}")
print(f"  PCA 保留成分數: {X_pca.shape[1]}")
print(f"  解釋變異量: {pca.explained_variance_ratio_.sum()*100:.1f}%")
print(f"  各成分解釋量: {[f'{v*100:.1f}%' for v in pca.explained_variance_ratio_]}")

# ─────────────────────────────────────────────────────────
# 4. K-Means: Elbow + Silhouette 自動選 k
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 | K-Means 分群 (Elbow + Silhouette)")
print("=" * 60)

K_RANGE = range(2, 11)
inertias = []
sil_scores = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
    labels_k = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_pca, labels_k, sample_size=min(30000, len(X_pca)))
    sil_scores.append(sil)
    print(f"  k={k:2d}: Inertia={km.inertia_:,.0f}, Silhouette={sil:.4f}")

# 自動選 k: Silhouette 最高
best_k = list(K_RANGE)[np.argmax(sil_scores)]
print(f"\n  ★ Silhouette 最高 k = {best_k} (score = {max(sil_scores):.4f})")

# 若 best_k=2 太粗略，取第二高且 k>=3
if best_k == 2:
    sil_copy = list(sil_scores)
    sil_copy[0] = -1  # 排除 k=2
    alt_k = list(K_RANGE)[np.argmax(sil_copy)]
    print(f"  → k=2 過於粗略，改用次佳 k={alt_k} (score={sil_copy[np.argmax(sil_copy)]:.4f})")
    best_k = alt_k

# 最終分群
print(f"\n  最終分群 k = {best_k}")
km_final = KMeans(n_clusters=best_k, n_init=20, max_iter=500, random_state=42)
labels = km_final.fit_predict(X_pca)

feat["cluster"] = labels

print(f"\n  各群樣本數:")
for c in range(best_k):
    cnt = (labels == c).sum()
    pct = cnt / len(labels) * 100
    print(f"    群 {c}: {cnt:>8,} 筆 ({pct:5.1f}%)")

# ─────────────────────────────────────────────────────────
# 5. 各群特徵描述
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 | 各群典型天氣模式描述")
print("=" * 60)

cluster_stats = feat.groupby("cluster").agg(["mean", "std", "median"])
# 扁平化欄位名
cluster_stats.columns = ['_'.join(col) for col in cluster_stats.columns]

# 計算各群的風向角度（從 sin/cos）
def describe_cluster(c, stats_row):
    """根據統計量自動產生天氣模式描述"""
    ws100 = stats_row.get("WS_100E_mean", 0)
    ws69  = stats_row.get("WS_69W_mean", 0)
    ti100 = stats_row.get("TI_100E_mean", 0)
    shear = stats_row.get("shear_38_69_mean", 0)
    temp  = stats_row.get("AT_95_mean", 0)
    pres  = stats_row.get("BP_93_mean", 0)
    wd_sin = stats_row.get("WD_97_sin_mean", 0)
    wd_cos = stats_row.get("WD_97_cos_mean", 0)
    
    # 風向角度
    wd_deg = np.degrees(np.arctan2(wd_sin, wd_cos)) % 360
    
    # 風向名稱
    dirs = ["北", "東北", "東", "東南", "南", "西南", "西", "西北"]
    dir_idx = int((wd_deg + 22.5) / 45) % 8
    dir_name = dirs[dir_idx]
    
    # 風速等級
    if ws100 < 3:
        ws_level = "微風"
    elif ws100 < 6:
        ws_level = "輕風"
    elif ws100 < 10:
        ws_level = "中等風速"
    elif ws100 < 15:
        ws_level = "強風"
    else:
        ws_level = "極強風"
    
    # 亂流描述
    if ti100 < 0.10:
        ti_desc = "低亂流(穩定)"
    elif ti100 < 0.20:
        ti_desc = "中等亂流"
    else:
        ti_desc = "高亂流(擾動)"
    
    # 風切描述
    if shear < 0.10:
        shear_desc = "低風切"
    elif shear < 0.20:
        shear_desc = "正常風切"
    elif shear < 0.35:
        shear_desc = "高風切"
    else:
        shear_desc = "極端風切"
    
    desc = f"{dir_name}風 {ws_level} | {ti_desc} | {shear_desc}"
    
    print(f"\n  群 {c}: {desc}")
    print(f"    100m 風速: {ws100:.2f} m/s (±{stats_row.get('WS_100E_std', 0):.2f})")
    print(f"    69m 風速:  {ws69:.2f} m/s")
    print(f"    亂流強度:  {ti100:.3f}")
    print(f"    風切指數:  {shear:.3f}")
    print(f"    風向:      {wd_deg:.0f}° ({dir_name})")
    print(f"    氣溫:      {temp:.1f}°C")
    print(f"    氣壓:      {pres:.1f} hPa")
    
    return desc

cluster_descriptions = {}
for c in range(best_k):
    row = {}
    for col in feat.columns:
        if col == "cluster":
            continue
        key_mean = f"{col}_mean"
        key_std  = f"{col}_std"
        if key_mean in cluster_stats.columns:
            row[key_mean] = cluster_stats.loc[c, key_mean]
        if key_std in cluster_stats.columns:
            row[key_std] = cluster_stats.loc[c, key_std]
    cluster_descriptions[c] = describe_cluster(c, row)

# ─────────────────────────────────────────────────────────
# 6. 視覺化（8 張子圖）
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 | 繪製分群分析圖表 (8 張子圖)")
print("=" * 60)

# 配色系統（與 wind_rf_regression.py 一致基底 + 群組色彩）
C_BG     = "#FAFBFC"
C_GRAY   = "#6B7280"

# 群組色彩 (鮮明可區分)
CLUSTER_COLORS = [
    "#0EA5E9",  # 天藍
    "#F97316",  # 橘
    "#10B981",  # 翠綠
    "#EF4444",  # 紅
    "#8B5CF6",  # 紫
    "#EC4899",  # 粉紅
    "#F59E0B",  # 金黃
    "#06B6D4",  # 青
    "#84CC16",  # 萊姆綠
]

fig = plt.figure(figsize=(18, 22), facecolor=C_BG)
fig.suptitle(
    "風況分群分析 — 典型天氣模式識別",
    fontsize=19, fontweight="bold", y=0.98, color="#1F2937"
)

# ── 6a. Elbow + Silhouette 雙軸圖 ──
ax1 = fig.add_subplot(4, 2, 1)
ax1.set_facecolor(C_BG)
ax1_twin = ax1.twinx()

ax1.plot(list(K_RANGE), inertias, 'o-', color="#0EA5E9", linewidth=2, markersize=6, label="Inertia")
ax1_twin.plot(list(K_RANGE), sil_scores, 's-', color="#F97316", linewidth=2, markersize=6, label="Silhouette")
ax1.axvline(best_k, color="#EF4444", linestyle="--", alpha=0.7, linewidth=1.5, label=f"最佳 k={best_k}")

ax1.set_xlabel("群數 k", fontsize=11)
ax1.set_ylabel("Inertia (SSE)", fontsize=11, color="#0EA5E9")
ax1_twin.set_ylabel("Silhouette Score", fontsize=11, color="#F97316")
ax1.set_title("(a) Elbow + Silhouette 分析", fontsize=13, fontweight="bold")
ax1.set_xticks(list(K_RANGE))
ax1.grid(True, alpha=0.15)

# 合併圖例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

# ── 6b. PCA 2D 散布圖 ──
ax2 = fig.add_subplot(4, 2, 2)
ax2.set_facecolor(C_BG)

for c in range(best_k):
    mask = labels == c
    ax2.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        s=2, alpha=0.15, c=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
        label=f"群{c}", edgecolors="none"
    )

# 畫群中心
centers_pca = km_final.cluster_centers_[:, :2]
for c in range(best_k):
    ax2.scatter(
        centers_pca[c, 0], centers_pca[c, 1],
        s=200, c=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
        marker="*", edgecolors="white", linewidths=1.5, zorder=10
    )

ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
ax2.set_title("(b) PCA 空間分群散布圖", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9, markerscale=5, loc="upper right")
ax2.grid(True, alpha=0.15)

# ── 6c. 各群風速箱型圖 ──
ax3 = fig.add_subplot(4, 2, 3)
ax3.set_facecolor(C_BG)

box_data = [feat.loc[feat["cluster"] == c, "WS_100E"].values for c in range(best_k)]
bp = ax3.boxplot(
    box_data, patch_artist=True, notch=True, widths=0.5,
    showfliers=False,
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(color=C_GRAY),
    capprops=dict(color=C_GRAY),
)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(CLUSTER_COLORS[i % len(CLUSTER_COLORS)])
    patch.set_alpha(0.8)
    patch.set_edgecolor("white")

ax3.set_xticklabels([f"群{c}" for c in range(best_k)], fontsize=10)
ax3.set_ylabel("100m 風速 (m/s)", fontsize=11)
ax3.set_title("(c) 各群 100m 風速分佈", fontsize=13, fontweight="bold")
ax3.grid(True, alpha=0.15, axis="y")

# 標註中位數
for c in range(best_k):
    median_val = np.median(box_data[c])
    ax3.text(c + 1, median_val + 0.3, f"{median_val:.1f}",
             ha="center", fontsize=9, fontweight="bold", color="#374151")

# ── 6d. 各群風向極座標圖 ──
ax4 = fig.add_subplot(4, 2, 4, projection="polar")
ax4.set_facecolor(C_BG)

theta_bins = np.linspace(0, 2 * np.pi, 17)  # 16 方位

for c in range(best_k):
    mask_c = feat["cluster"] == c
    wd_sin_c = feat.loc[mask_c, "WD_97_sin"]
    wd_cos_c = feat.loc[mask_c, "WD_97_cos"]
    # 計算角度 (氣象慣例: 北=0, 順時針)
    wd_rad = np.arctan2(wd_sin_c, wd_cos_c)
    # 轉換: 數學角 → 氣象角 (北=0 → 90-θ), 再轉回弧度
    wd_met = (np.pi/2 - wd_rad) % (2 * np.pi)
    
    counts, _ = np.histogram(wd_met, bins=theta_bins)
    counts = counts / counts.sum()  # 正規化
    
    # 繪製
    theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
    width = 2 * np.pi / 16 * 0.7
    ax4.bar(
        theta_centers, counts, width=width,
        color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
        alpha=0.5, label=f"群{c}", edgecolor="white", linewidth=0.3
    )

ax4.set_theta_zero_location("N")
ax4.set_theta_direction(-1)  # 順時針
ax4.set_title("(d) 各群風向分佈", fontsize=13, fontweight="bold", pad=20)
ax4.legend(fontsize=8, loc="lower left", bbox_to_anchor=(-0.15, -0.15))

# ── 6e. 各群特徵雷達圖 ──
ax5 = fig.add_subplot(4, 2, 5, projection="polar")

radar_features = ["WS_100E", "WS_69W", "WS_38W", "TI_100E", "shear_38_69", "AT_95", "BP_93"]
radar_labels   = ["100m風速", "69m風速", "38m風速", "亂流強度", "風切指數", "氣溫", "氣壓"]

# 正規化到 0-1
radar_data = {}
for col in radar_features:
    col_min = feat[col].min()
    col_max = feat[col].max()
    for c in range(best_k):
        if c not in radar_data:
            radar_data[c] = []
        val = feat.loc[feat["cluster"] == c, col].mean()
        radar_data[c].append((val - col_min) / (col_max - col_min + 1e-9))

angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
angles += angles[:1]  # 閉合

for c in range(best_k):
    values = radar_data[c] + radar_data[c][:1]
    ax5.plot(angles, values, 'o-', color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
             linewidth=2, markersize=4, label=f"群{c}")
    ax5.fill(angles, values, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], alpha=0.1)

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(radar_labels, fontsize=9)
ax5.set_title("(e) 各群特徵雷達圖 (正規化)", fontsize=13, fontweight="bold", pad=20)
ax5.legend(fontsize=8, loc="lower left", bbox_to_anchor=(-0.2, -0.15))
ax5.set_ylim(0, 1)
ax5.grid(True, alpha=0.3)

# ── 6f. 各群日內分佈 ──
ax6 = fig.add_subplot(4, 2, 6)
ax6.set_facecolor(C_BG)

for c in range(best_k):
    mask_c = feat["cluster"] == c
    hours = feat.index[mask_c].hour
    hour_counts = np.bincount(hours, minlength=24)
    hour_pct = hour_counts / hour_counts.sum()
    ax6.plot(range(24), hour_pct, 'o-', color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
             linewidth=2, markersize=4, label=f"群{c}", alpha=0.9)

ax6.set_xlabel("小時 (Local Time)", fontsize=11)
ax6.set_ylabel("佔比", fontsize=11)
ax6.set_title("(f) 各群日內分佈", fontsize=13, fontweight="bold")
ax6.set_xticks(range(0, 24, 3))
ax6.set_xlim(-0.5, 23.5)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.15)

# ── 6g. 各群月份分佈熱力圖 ──
ax7 = fig.add_subplot(4, 2, 7)
ax7.set_facecolor(C_BG)

month_cluster = pd.crosstab(feat.index.month, feat["cluster"], normalize="columns")
month_labels = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]

sns.heatmap(
    month_cluster, ax=ax7, cmap="YlOrRd", annot=True, fmt=".2f",
    xticklabels=[f"群{c}" for c in range(best_k)],
    yticklabels=month_labels,
    linewidths=0.5, linecolor="white",
    cbar_kws={"shrink": 0.8, "label": "佔比"}
)
ax7.set_title("(g) 各群月份分佈 (行內正規化)", fontsize=13, fontweight="bold")
ax7.set_ylabel("月份", fontsize=11)
ax7.set_xlabel("")

# ── 6h. 月份堆疊佔比圖 ──
ax8 = fig.add_subplot(4, 2, 8)
ax8.set_facecolor(C_BG)

# 按月彙總各群佔比
feat["year_month"] = feat.index.to_period("M")
month_pivot = pd.crosstab(feat["year_month"], feat["cluster"], normalize="index")

# 堆疊面積圖
x_vals = range(len(month_pivot))
bottom = np.zeros(len(month_pivot))

for c in range(best_k):
    if c in month_pivot.columns:
        values = month_pivot[c].values
        ax8.fill_between(x_vals, bottom, bottom + values,
                         color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                         alpha=0.75, label=f"群{c}")
        bottom += values

# x 軸標籤 (每 6 個月顯示)
tick_positions = list(range(0, len(month_pivot), 6))
tick_labels = [str(month_pivot.index[i]) for i in tick_positions if i < len(month_pivot)]
ax8.set_xticks(tick_positions[:len(tick_labels)])
ax8.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
ax8.set_ylabel("佔比", fontsize=11)
ax8.set_title("(h) 各群月度佔比變化", fontsize=13, fontweight="bold")
ax8.legend(fontsize=9, loc="upper right")
ax8.set_xlim(0, len(month_pivot) - 1)
ax8.set_ylim(0, 1)
ax8.grid(True, alpha=0.15, axis="y")

# 清理暫時欄位
feat.drop(columns=["year_month"], inplace=True)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# 儲存
output_path = Path(OUTPUT_DIR) / "wind_clustering_results.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"\n✓ 圖表已儲存: {output_path}")

plt.show(block=False)
plt.pause(1)
plt.close("all")

# ─────────────────────────────────────────────────────────
# 7. 匯出結果
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 | 匯出結果")
print("=" * 60)

# 7a. 分群統計摘要
summary_rows = []
for c in range(best_k):
    mask_c = feat["cluster"] == c
    row = {
        "cluster": c,
        "description": cluster_descriptions[c],
        "count": mask_c.sum(),
        "pct": f"{mask_c.sum()/len(feat)*100:.1f}%",
        "WS_100E_mean": feat.loc[mask_c, "WS_100E"].mean(),
        "WS_100E_median": feat.loc[mask_c, "WS_100E"].median(),
        "WS_100E_std": feat.loc[mask_c, "WS_100E"].std(),
        "WS_69W_mean": feat.loc[mask_c, "WS_69W"].mean(),
        "WS_38W_mean": feat.loc[mask_c, "WS_38W"].mean(),
        "TI_100E_mean": feat.loc[mask_c, "TI_100E"].mean(),
        "shear_38_69_mean": feat.loc[mask_c, "shear_38_69"].mean(),
        "AT_95_mean": feat.loc[mask_c, "AT_95"].mean(),
        "BP_93_mean": feat.loc[mask_c, "BP_93"].mean(),
    }
    # 主要風向
    wd_sin_c = feat.loc[mask_c, "WD_97_sin"].mean()
    wd_cos_c = feat.loc[mask_c, "WD_97_cos"].mean()
    wd_deg = np.degrees(np.arctan2(wd_sin_c, wd_cos_c)) % 360
    row["WD_97_deg"] = wd_deg
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_path = Path(OUTPUT_DIR) / "wind_clustering_summary.csv"
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
print(f"✓ 分群摘要已匯出: {summary_path}")

# 7b. 含群標籤的完整資料
labeled_path = Path(OUTPUT_DIR) / "wind_clustering_labeled.csv"
feat.to_csv(labeled_path, encoding="utf-8-sig")
print(f"✓ 標籤資料已匯出: {labeled_path}")
print(f"  共 {len(feat):,} 筆")

# ─────────────────────────────────────────────────────────
# 8. 最終摘要
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("最終摘要")
print("=" * 60)

print(f"""
  方法:     10-min 彙總 → StandardScaler → PCA → K-Means
  特徵數:   {feat.shape[1] - 1} 個 (不含 cluster 標籤)
  樣本數:   {len(feat):,} 筆
  PCA 維度: {X_pca.shape[1]} 維 (解釋 {pca.explained_variance_ratio_.sum()*100:.1f}%)
  最佳 k:   {best_k} (Silhouette = {max(sil_scores):.4f})

  ┌─────────────────────────────────────────────┐
  │  各群典型天氣模式                            │""")

for c in range(best_k):
    mask_c = feat["cluster"] == c
    cnt = mask_c.sum()
    pct = cnt / len(feat) * 100
    print(f"  │  群{c}: {cluster_descriptions[c]:<36s} │")
    print(f"  │       {cnt:>8,} 筆 ({pct:4.1f}%)                     │")

print(f"""  └─────────────────────────────────────────────┘

  輸出檔案:
    📊 {output_path.name}
    📄 {summary_path.name}
    📄 {labeled_path.name}
""")

print("Done! ✓")
