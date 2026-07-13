"""
============================================================
  風況分類 — 低風 / 正常 / 強風 / 極端
  Wind Regime Classification with Confusion Matrix

  資料: BSMI 測風塔 1-min 資料 (parquet) → 10-min 彙總
  方法: 風速閾值定義標籤 → 特徵工程 → Random Forest → 混淆矩陣

  分類定義 (基於 100m 風速):
    低風   : WS < 4 m/s
    正常   : 4 ≤ WS < 10 m/s
    強風   : 10 ≤ WS < 20 m/s
    極端   : WS ≥ 20 m/s

  執行方式:
    pip install pandas pyarrow numpy scikit-learn matplotlib seaborn
    python wind_classification.py
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# 本地路徑設定
# ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'data', 'BSMI_wind_1min_parquet'))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')

# ─────────────────────────────────────────────────────────
# 0. 設定中文字型
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
# 風況分類閾值定義
# ─────────────────────────────────────────────────────────
WIND_CLASSES = {
    "低風":  (0,    4),
    "正常":  (4,   10),
    "強風":  (10,  20),
    "極端":  (20, 999),
}
CLASS_ORDER = ["低風", "正常", "強風", "極端"]
CLASS_COLORS = {
    "低風": "#0EA5E9",   # 天藍 — 平靜
    "正常": "#10B981",   # 翠綠 — 適宜
    "強風": "#F97316",   # 橘   — 警示
    "極端": "#EF4444",   # 紅   — 危險
}

def assign_wind_class(ws):
    """根據 100m 風速指定風況標籤"""
    for label, (lo, hi) in WIND_CLASSES.items():
        if lo <= ws < hi:
            return label
    return "極端"

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

df_raw = pd.concat(dfs).sort_index()
print(f"  合併後: {df_raw.shape[0]:,} 筆, {df_raw.shape[1]} 欄")
print(f"  期間: {df_raw.index.min()} ~ {df_raw.index.max()}")

# ─────────────────────────────────────────────────────────
# 2. 10-min 彙總 + 特徵工程
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 | 10-min 彙總 + 特徵工程")
print("=" * 60)

agg_dict = {
    'WS_100E': ['mean', 'std', 'max', 'min'],
    'WS_100W': ['mean'],
    'WS_69W':  ['mean', 'std'],
    'WS_38W':  ['mean', 'std'],
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

# ── 建立標籤 (y) ──
y_labels = df_10["WS_100E_mean"].apply(assign_wind_class)

# ── 建立特徵 (X) ── 故意排除 WS_100E_mean 作為主要特徵，
# 以模擬「根據其他觀測量推斷風況等級」的分類任務
feat = pd.DataFrame(index=df_10.index)

# 其他高度風速（作為間接指標）
feat["WS_100W"]    = df_10["WS_100W_mean"]
feat["WS_69W"]     = df_10["WS_69W_mean"]
feat["WS_38W"]     = df_10["WS_38W_mean"]

# 10-min 統計特徵
feat["WS_100E_std"] = df_10["WS_100E_std"]
feat["WS_100E_max"] = df_10["WS_100E_max"]
feat["WS_100E_min"] = df_10["WS_100E_min"]
feat["WS_100E_range"] = df_10["WS_100E_max"] - df_10["WS_100E_min"]
feat["WS_69W_std"]  = df_10["WS_69W_std"]
feat["WS_38W_std"]  = df_10["WS_38W_std"]

# 亂流強度
feat["TI_100E"] = df_10["WS_100E_std"] / df_10["WS_100E_mean"].clip(lower=0.5)
feat["TI_69W"]  = df_10["WS_69W_std"]  / df_10["WS_69W_mean"].clip(lower=0.5)

# 風切指數
with np.errstate(divide="ignore", invalid="ignore"):
    feat["shear_38_69"] = (
        np.log(df_10["WS_69W_mean"].clip(0.3) / df_10["WS_38W_mean"].clip(0.3))
        / np.log(69 / 38)
    )
    feat["shear_69_100"] = (
        np.log(df_10["WS_100E_mean"].clip(0.3) / df_10["WS_69W_mean"].clip(0.3))
        / np.log(100 / 69)
    )
feat["shear_38_69"]  = feat["shear_38_69"].clip(-1, 3)
feat["shear_69_100"] = feat["shear_69_100"].clip(-1, 3)

# 風速比例
feat["ratio_69_38"]  = df_10["WS_69W_mean"] / df_10["WS_38W_mean"].clip(lower=0.3)
feat["ratio_100_69"] = df_10["WS_100E_mean"] / df_10["WS_69W_mean"].clip(lower=0.3)

# 風向 (sin/cos 向量)
feat["WD_97_sin"] = df_10["WD_97_sin_mean"]
feat["WD_97_cos"] = df_10["WD_97_cos_mean"]
feat["WD_35_sin"] = df_10["WD_35_sin_mean"]
feat["WD_35_cos"] = df_10["WD_35_cos_mean"]

# 氣象共變量
feat["AT_95"] = df_10["AT_95_mean"]
feat["RH_95"] = df_10["RH_95_mean"]
feat["BP_93"] = df_10["BP_93_mean"]

# 日週期 & 年週期
feat["hour_sin"] = np.sin(2 * np.pi * df_10.index.hour / 24)
feat["hour_cos"] = np.cos(2 * np.pi * df_10.index.hour / 24)
feat["month_sin"] = np.sin(2 * np.pi * df_10.index.month / 12)
feat["month_cos"] = np.cos(2 * np.pi * df_10.index.month / 12)

# 清除殘餘 NaN / inf
feat = feat.replace([np.inf, -np.inf], np.nan)
valid_mask = feat.notna().all(axis=1) & y_labels.notna()
feat = feat.loc[valid_mask]
y_labels = y_labels.loc[valid_mask]

print(f"  特徵數: {feat.shape[1]} 個")
print(f"  有效樣本: {feat.shape[0]:,} 筆")

# 各類別統計
print("\n  風況分佈:")
class_counts = y_labels.value_counts()
for cls in CLASS_ORDER:
    cnt = class_counts.get(cls, 0)
    pct = cnt / len(y_labels) * 100
    print(f"    {cls}: {cnt:>8,} 筆 ({pct:5.1f}%)")

# ─────────────────────────────────────────────────────────
# 3. 訓練 / 測試分割
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 | 訓練 / 測試分割")
print("=" * 60)

le = LabelEncoder()
le.fit(CLASS_ORDER)
y_encoded = le.transform(y_labels)

X_train, X_test, y_train, y_test = train_test_split(
    feat.values, y_encoded,
    test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"  訓練集: {len(y_train):,} 筆")
print(f"  測試集: {len(y_test):,} 筆")
print(f"  訓練集類別比例: {dict(zip(le.inverse_transform(np.unique(y_train)), np.bincount(y_train)))}")

# ─────────────────────────────────────────────────────────
# 4. Random Forest 分類器訓練
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 | 訓練 Random Forest 分類器")
print("=" * 60)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced",  # 處理類別不平衡
    random_state=42,
    n_jobs=-1,
)

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test  = clf.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc  = accuracy_score(y_test,  y_pred_test)
train_f1  = f1_score(y_train, y_pred_train, average="weighted")
test_f1   = f1_score(y_test,  y_pred_test,  average="weighted")

print(f"  訓練集 Accuracy: {train_acc:.4f}")
print(f"  測試集 Accuracy: {test_acc:.4f}")
print(f"  訓練集 F1 (加權): {train_f1:.4f}")
print(f"  測試集 F1 (加權): {test_f1:.4f}")

# ─────────────────────────────────────────────────────────
# 5. 分類報告 + 混淆矩陣
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 | 分類結果評估")
print("=" * 60)

target_names = CLASS_ORDER
print("\n📋 分類報告 (測試集):")
print(classification_report(
    y_test, y_pred_test,
    target_names=target_names,
    digits=4
))

cm = confusion_matrix(y_test, y_pred_test)
cm_normalized = confusion_matrix(y_test, y_pred_test, normalize="true")

print("📊 混淆矩陣 (原始計數):")
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
print(cm_df.to_string())

print("\n📊 混淆矩陣 (正規化):")
cm_norm_df = pd.DataFrame(
    np.round(cm_normalized, 4), index=target_names, columns=target_names
)
print(cm_norm_df.to_string())

# 特徵重要性
feat_importance = pd.DataFrame({
    "feature": feat.columns,
    "importance": clf.feature_importances_
}).sort_values("importance", ascending=False)

print("\n🔑 Top-10 重要特徵:")
for i, row in feat_importance.head(10).iterrows():
    print(f"  {row['feature']:<20s} {row['importance']:.4f}")

# ─────────────────────────────────────────────────────────
# 6. 視覺化 (6 張子圖)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 | 繪製風況分類分析圖 (6 張子圖)")
print("=" * 60)

C_BG     = "#FAFBFC"
C_TEXT   = "#1F2937"
C_GRAY   = "#6B7280"

fig = plt.figure(figsize=(18, 24), facecolor=C_BG)
fig.suptitle(
    "風況分類分析 — 低風 / 正常 / 強風 / 極端",
    fontsize=20, fontweight="bold", y=0.985, color=C_TEXT
)

# ── 6a. 混淆矩陣 (計數) ──
ax1 = fig.add_subplot(3, 2, 1)
ax1.set_facecolor(C_BG)

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=target_names, yticklabels=target_names,
    linewidths=1.5, linecolor="white",
    ax=ax1,
    annot_kws={"fontsize": 13, "fontweight": "bold"},
    cbar_kws={"shrink": 0.8},
)
ax1.set_xlabel("預測類別", fontsize=12, fontweight="bold")
ax1.set_ylabel("真實類別", fontsize=12, fontweight="bold")
ax1.set_title(
    f"(a) 混淆矩陣 — 計數  |  Accuracy={test_acc:.2%}",
    fontsize=14, fontweight="bold", pad=12
)
ax1.tick_params(axis='both', labelsize=11)

# ── 6b. 混淆矩陣 (正規化 %) ──
ax2 = fig.add_subplot(3, 2, 2)
ax2.set_facecolor(C_BG)

sns.heatmap(
    cm_normalized, annot=True, fmt=".2%", cmap="YlOrRd",
    xticklabels=target_names, yticklabels=target_names,
    linewidths=1.5, linecolor="white", vmin=0, vmax=1,
    ax=ax2,
    annot_kws={"fontsize": 13, "fontweight": "bold"},
    cbar_kws={"shrink": 0.8, "label": "比例"},
)
ax2.set_xlabel("預測類別", fontsize=12, fontweight="bold")
ax2.set_ylabel("真實類別", fontsize=12, fontweight="bold")
ax2.set_title("(b) 混淆矩陣 — 正規化 (Recall %)", fontsize=14, fontweight="bold", pad=12)
ax2.tick_params(axis='both', labelsize=11)

# ── 6c. 各類別風速分佈 (violin + strip) ──
ax3 = fig.add_subplot(3, 2, 3)
ax3.set_facecolor(C_BG)

# 為 violin plot 準備資料
plot_df = pd.DataFrame({
    "WS_100E": df_10.loc[feat.index, "WS_100E_mean"],
    "風況": y_labels,
})
# 確保類別順序
plot_df["風況"] = pd.Categorical(plot_df["風況"], categories=CLASS_ORDER, ordered=True)

palette = [CLASS_COLORS[c] for c in CLASS_ORDER]
parts = ax3.violinplot(
    [plot_df.loc[plot_df["風況"] == c, "WS_100E"].values for c in CLASS_ORDER],
    positions=range(len(CLASS_ORDER)),
    showmeans=True, showmedians=True, showextrema=False,
)
# 上色
for i, body in enumerate(parts['bodies']):
    body.set_facecolor(palette[i])
    body.set_alpha(0.6)
    body.set_edgecolor("white")
    body.set_linewidth(1)

parts['cmeans'].set_color('#1F2937')
parts['cmeans'].set_linewidth(2)
parts['cmedians'].set_color('#EF4444')
parts['cmedians'].set_linewidth(2)

# 標註統計量
for i, cls in enumerate(CLASS_ORDER):
    subset = plot_df.loc[plot_df["風況"] == cls, "WS_100E"]
    cnt = len(subset)
    med = subset.median()
    ax3.text(i, med + 0.8, f"n={cnt:,}\nmed={med:.1f}",
             ha="center", va="bottom", fontsize=8, fontweight="bold", color=C_TEXT)

# 閾值線
for threshold, label in [(4, "4 m/s"), (10, "10 m/s"), (20, "20 m/s")]:
    ax3.axhline(threshold, color="#9CA3AF", linestyle="--", linewidth=1, alpha=0.7)
    ax3.text(len(CLASS_ORDER) - 0.5, threshold + 0.3, label,
             fontsize=8, color="#6B7280", ha="right")

ax3.set_xticks(range(len(CLASS_ORDER)))
ax3.set_xticklabels(CLASS_ORDER, fontsize=11)
ax3.set_ylabel("100m 風速 (m/s)", fontsize=12)
ax3.set_title("(c) 各風況類別的 100m 風速分佈", fontsize=14, fontweight="bold", pad=12)
ax3.grid(True, alpha=0.15, axis="y")

# ── 6d. 特徵重要性 (Top 15) ──
ax4 = fig.add_subplot(3, 2, 4)
ax4.set_facecolor(C_BG)

top_n = 15
top_feat = feat_importance.head(top_n).iloc[::-1]  # 反轉讓最重要的在上方
colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))[::-1]

bars = ax4.barh(
    range(top_n), top_feat["importance"].values,
    color=colors_bar, edgecolor="white", linewidth=0.8, height=0.7
)

ax4.set_yticks(range(top_n))
ax4.set_yticklabels(top_feat["feature"].values, fontsize=10)
ax4.set_xlabel("重要性 (Gini Importance)", fontsize=12)
ax4.set_title("(d) 特徵重要性 Top-15", fontsize=14, fontweight="bold", pad=12)
ax4.grid(True, alpha=0.15, axis="x")

# 標註數值
for bar, val in zip(bars, top_feat["importance"].values):
    ax4.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=9, color=C_TEXT)

# ── 6e. 各類別 Precision / Recall / F1 長條圖 ──
ax5 = fig.add_subplot(3, 2, 5)
ax5.set_facecolor(C_BG)

report_dict = classification_report(
    y_test, y_pred_test, target_names=target_names, output_dict=True
)

x_pos = np.arange(len(CLASS_ORDER))
width = 0.25
metrics = ["precision", "recall", "f1-score"]
metric_colors = ["#0EA5E9", "#10B981", "#F97316"]
metric_labels = ["Precision", "Recall", "F1-Score"]

for i, (metric, color, label) in enumerate(zip(metrics, metric_colors, metric_labels)):
    values = [report_dict[cls][metric] for cls in CLASS_ORDER]
    bars = ax5.bar(x_pos + i * width, values, width,
                   color=color, alpha=0.85, label=label, edgecolor="white", linewidth=0.8)
    # 標註數值
    for bar, val in zip(bars, values):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8.5,
                 fontweight="bold", color=C_TEXT)

ax5.set_xticks(x_pos + width)
ax5.set_xticklabels(CLASS_ORDER, fontsize=11)
ax5.set_ylabel("Score", fontsize=12)
ax5.set_ylim(0, 1.15)
ax5.set_title("(e) 各類別 Precision / Recall / F1", fontsize=14, fontweight="bold", pad=12)
ax5.legend(fontsize=10, loc="upper right")
ax5.grid(True, alpha=0.15, axis="y")
ax5.axhline(1.0, color="#D1D5DB", linestyle=":", linewidth=1)

# ── 6f. 月份 × 類別 堆疊比例圖 ──
ax6 = fig.add_subplot(3, 2, 6)
ax6.set_facecolor(C_BG)

month_class = pd.crosstab(y_labels.index.month, y_labels, normalize="index")
# 確保欄位順序
for cls in CLASS_ORDER:
    if cls not in month_class.columns:
        month_class[cls] = 0
month_class = month_class[CLASS_ORDER]

months = range(1, 13)
bottom_stack = np.zeros(12)
month_labels = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]

for cls in CLASS_ORDER:
    values = month_class[cls].values
    ax6.bar(months, values, bottom=bottom_stack,
            color=CLASS_COLORS[cls], alpha=0.85, label=cls,
            edgecolor="white", linewidth=0.5, width=0.8)
    # 標註比例 (只標 > 5%)
    for m, (v, b) in enumerate(zip(values, bottom_stack)):
        if v > 0.05:
            ax6.text(m + 1, b + v / 2, f"{v:.0%}",
                     ha="center", va="center", fontsize=7.5,
                     fontweight="bold", color="white")
    bottom_stack += values

ax6.set_xticks(months)
ax6.set_xticklabels(month_labels, fontsize=10)
ax6.set_ylabel("佔比", fontsize=12)
ax6.set_ylim(0, 1)
ax6.set_title("(f) 各月份風況類別佔比", fontsize=14, fontweight="bold", pad=12)
ax6.legend(fontsize=10, loc="upper right")
ax6.grid(True, alpha=0.15, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.97])

# 儲存
output_path = Path(OUTPUT_DIR) / "wind_classification_results.png"
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

# 7a. 分類結果 CSV
result_df = pd.DataFrame(index=feat.index)
result_df["WS_100E_mean"] = df_10.loc[feat.index, "WS_100E_mean"]
result_df["wind_class_true"] = y_labels

# 全量預測
y_all_pred = clf.predict(feat.values)
result_df["wind_class_pred"] = le.inverse_transform(y_all_pred)
result_df["correct"] = result_df["wind_class_true"] == result_df["wind_class_pred"]

result_path = Path(OUTPUT_DIR) / "wind_classification_labeled.csv"
result_df.to_csv(result_path, encoding="utf-8-sig")
print(f"✓ 標籤資料已匯出: {result_path}")
print(f"  共 {len(result_df):,} 筆")

# 7b. 混淆矩陣 CSV
cm_full_df = pd.DataFrame(cm, index=target_names, columns=target_names)
cm_path = Path(OUTPUT_DIR) / "wind_classification_confusion_matrix.csv"
cm_full_df.to_csv(cm_path, encoding="utf-8-sig")
print(f"✓ 混淆矩陣已匯出: {cm_path}")

# 7c. 分類報告 CSV
report_df = pd.DataFrame(report_dict).T
report_path = Path(OUTPUT_DIR) / "wind_classification_report.csv"
report_df.to_csv(report_path, encoding="utf-8-sig")
print(f"✓ 分類報告已匯出: {report_path}")

# ─────────────────────────────────────────────────────────
# 8. 最終摘要
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("最終摘要")
print("=" * 60)

print(f"""
  任務:     風況四級分類 (低風/正常/強風/極端)
  方法:     10-min 彙總 → 特徵工程 → Random Forest Classifier
  特徵數:   {feat.shape[1]} 個
  樣本數:   {len(feat):,} 筆 (80/20 訓練/測試)
  模型:     Random Forest (n_estimators=300, max_depth=20)

  ┌──────────────────────────────────────────────────┐
  │  測試集效能                                       │
  │  Accuracy:    {test_acc:.4f}                              │
  │  F1 (加權):   {test_f1:.4f}                              │
  └──────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────┐
  │  風況閾值定義                                     │
  │  低風:   WS < 4 m/s                              │
  │  正常:   4 ≤ WS < 10 m/s                         │
  │  強風:   10 ≤ WS < 20 m/s                        │
  │  極端:   WS ≥ 20 m/s                             │
  └──────────────────────────────────────────────────┘

  輸出檔案:
    📊 {output_path.name}
    📄 {result_path.name}
    📄 {cm_path.name}
    📄 {report_path.name}
""")

print("Done! ✓")
