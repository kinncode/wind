#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BSMI 測風塔 2016-2021 侵台颱風風速與氣壓特徵分析
Typhoon Wind Speed & Barometric Pressure Analysis (2016-2021)

本程式包含：
1. 載入 2016-2021 年共 64 個月份的 Parquet 原始資料
2. 提取 100m 高空風速與 93m 大氣壓力資料
3. 針對 5 個著名侵台颱風日期區間進行高頻 (1-min) 時序特徵切片：
   - 2016 梅姬 (Megi)
   - 2017 尼莎 & 海棠 (Nesha & Haitang)
   - 2018 瑪莉亞 (Maria)
   - 2019 利奇馬 (Lekima)
   - 2021 璨樹 (Chanthu)
4. 繪製精緻的「雙 Y 軸」颱風過境對比圖 (風速暴增 vs 氣壓陡降)
5. 繪製歷年颱風季 (7-9月) 的風速與氣壓統計分布圖 (Violin Plot)，分析歷年強弱變化
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
            print(f"[OK] 使用字型: {font}")
            return font
    print("[WARN] 未找到中文字型，圖表文字可能顯示為方塊")
    plt.rcParams["axes.unicode_minus"] = False
    return None

FONT = setup_chinese_font()

# =====================================================================
# 1. 載入並彙整全量資料
# =====================================================================
print("\n" + "=" * 60)
print("STEP 1 | 載入並彙整 2016-2021 測風塔全量數據 (這可能需要 10-15 秒)")
print("=" * 60)

files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
if len(files) == 0:
    raise FileNotFoundError(f"在 {DATA_DIR} 中找不到任何 parquet 檔案")

dfs = []
for f in files:
    # 僅讀取時間索引與需要的欄位，以加速讀取並節省記憶體
    df_temp = pd.read_parquet(f, columns=['WS_100E', 'WS_100W', 'BP_93'])
    # 計算 100m 平均風速
    df_temp['WS_100'] = (df_temp['WS_100E'] + df_temp['WS_100W']) / 2.0
    df_temp = df_temp[['WS_100', 'BP_93']].dropna()
    dfs.append(df_temp)

df_all = pd.concat(dfs).sort_index()
print(f"  [OK] 合併成功！總數據筆數: {df_all.shape[0]:,} 筆 (1分鐘高頻資料)")
print(f"  [OK] 時間範圍: {df_all.index.min()} ~ {df_all.index.max()}")

# =====================================================================
# 2. 定義侵台颱風區間
# =====================================================================
typhoons = {
    "2016 梅姬 (Megi)": {
        "start": "2016-09-25 00:00:00",
        "end": "2016-09-29 00:00:00",
        "color_ws": "#0EA5E9",
        "color_bp": "#EF4444"
    },
    "2017 尼莎&海棠": {
        "start": "2017-07-28 00:00:00",
        "end": "2017-08-01 00:00:00",
        "color_ws": "#10B981",
        "color_bp": "#F97316"
    },
    "2018 瑪莉亞 (Maria)": {
        "start": "2018-07-09 00:00:00",
        "end": "2018-07-12 00:00:00",
        "color_ws": "#8B5CF6",
        "color_bp": "#EC4899"
    },
    "2019 利奇馬 (Lekima)": {
        "start": "2019-08-07 00:00:00",
        "end": "2019-08-11 00:00:00",
        "color_ws": "#06B6D4",
        "color_bp": "#F43F5E"
    },
    "2021 璨樹 (Chanthu)": {
        "start": "2021-09-10 00:00:00",
        "end": "2021-09-14 00:00:00",
        "color_ws": "#3B82F6",
        "color_bp": "#D946EF"
    }
}

# =====================================================================
# 3. 繪製 5 大颱風過境對比圖 (風速暴增 vs 氣壓陡降)
# =====================================================================
print("\n" + "=" * 60)
print("STEP 2 | 繪製 5 大侵台颱風過境高頻變化圖")
print("=" * 60)

fig, axes = plt.subplots(5, 1, figsize=(14, 18), facecolor='#FAFBFC')

for idx, (name, cfg) in enumerate(typhoons.items()):
    ax_ws = axes[idx]
    
    # 切片數據
    df_slice = df_all.loc[cfg["start"] : cfg["end"]]
    if len(df_slice) == 0:
        print(f"  ⚠ 找不到 {name} 期間的資料，跳過繪圖")
        continue
        
    # 進行 10-min 平滑以避免 1-min 噪訊過多，讓曲線更漂亮
    df_smooth = df_slice.resample('10min').mean().dropna()
    
    # 繪製左軸：風速
    line_ws = ax_ws.plot(df_smooth.index, df_smooth['WS_100'], color=cfg["color_ws"], label='100m 風速', lw=1.8)
    ax_ws.set_ylabel('風速 (m/s)', color=cfg["color_ws"], fontsize=11, fontweight='bold')
    ax_ws.tick_params(axis='y', labelcolor=cfg["color_ws"])
    ax_ws.grid(True, alpha=0.15)
    
    # 建立右軸：氣壓
    ax_bp = ax_ws.twinx()
    line_bp = ax_bp.plot(df_smooth.index, df_smooth['BP_93'], color=cfg["color_bp"], label='大氣壓力', linestyle='--', lw=1.5)
    ax_bp.set_ylabel('大氣壓力 (hPa)', color=cfg["color_bp"], fontsize=11, fontweight='bold')
    ax_bp.tick_params(axis='y', labelcolor=cfg["color_bp"])
    
    # 找出極大風速與極小氣壓
    max_ws = df_smooth['WS_100'].max()
    max_ws_t = df_smooth['WS_100'].idxmax()
    min_bp = df_smooth['BP_93'].min()
    min_bp_t = df_smooth['BP_93'].idxmin()
    
    # 標記極值點
    ax_ws.scatter(max_ws_t, max_ws, color='#EF4444', s=50, zorder=5)
    ax_ws.annotate(f"最大風速: {max_ws:.2f} m/s", xy=(max_ws_t, max_ws), xytext=(10, 10),
                    textcoords='offset points', arrowprops=dict(arrowstyle="->", color='#EF4444'), fontsize=9, fontweight='bold')
                    
    ax_bp.scatter(min_bp_t, min_bp, color='#3B82F6', s=50, zorder=5)
    ax_bp.annotate(f"最低氣壓: {min_bp:.1f} hPa", xy=(min_bp_t, min_bp), xytext=(-80, -15),
                    textcoords='offset points', arrowprops=dict(arrowstyle="->", color='#3B82F6'), fontsize=9, fontweight='bold')

    ax_ws.set_title(f"{name} 颱風過境對比 (期間: {cfg['start'][:10]} ~ {cfg['end'][:10]})", fontsize=13, fontweight='bold', pad=10)
    
    # X軸格式
    import matplotlib.dates as mdates
    ax_ws.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))

plt.suptitle('BSMI 測風塔 2016-2021 典型侵台颱風過境「風速與氣壓」高頻變化特徵對比', fontsize=17, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.98])

chart_path = os.path.join(OUTPUT_DIR, 'typhoon_high_freq_comparison.png')
plt.savefig(chart_path, dpi=150, facecolor='#FAFBFC')
plt.close()
print(f"  [OK] 5 大颱風高頻變化對比圖已儲存至: {chart_path}")

# =====================================================================
# 4. 繪製歷年颱風季 (7-9月) 統計分布圖 (Violin Plot / Box Plot)
# =====================================================================
print("\n" + "=" * 60)
print("STEP 3 | 分析歷年颱風季 (7-9月) 的整體變化規律")
print("=" * 60)

# 篩選歷年 7、8、9 月的數據
df_all['Month'] = df_all.index.month
df_all['Year'] = df_all.index.year

df_typhoon_season = df_all[df_all['Month'].isin([7, 8, 9])]

# 為了繪圖速度與清晰度，我們將資料 Resample 到 1 小時平均值進行分布統計
df_season_hourly = df_typhoon_season.resample('1h').mean().dropna()
df_season_hourly['Year'] = df_season_hourly.index.year
df_season_hourly['Year'] = df_season_hourly['Year'].astype(int)

# 繪圖
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), facecolor='#FAFBFC')

# 子圖 1: 風速 Violin Plot
sns.violinplot(x='Year', y='WS_100', data=df_season_hourly, ax=ax1, palette='Blues', inner='quartile')
ax1.set_title('歷年颱風季 (7-9月) 100m 高空風速分布對比 (2016-2021)', fontsize=13, fontweight='bold')
ax1.set_xlabel('年份', fontsize=11)
ax1.set_ylabel('小時平均風速 (m/s)', fontsize=11)
ax1.grid(axis='y', alpha=0.2)

# 子圖 2: 氣壓 Box Plot (氣壓的極小值代表颱風侵襲的劇烈程度)
sns.boxplot(x='Year', y='BP_93', data=df_season_hourly, ax=ax2, palette='Oranges')
ax2.set_title('歷年颱風季 (7-9月) 大氣壓力分布對比 (箱形圖離群點越低代表颱風越強)', fontsize=13, fontweight='bold')
ax2.set_xlabel('年份', fontsize=11)
ax2.set_ylabel('大氣壓力 (hPa)', fontsize=11)
ax2.grid(axis='y', alpha=0.2)

plt.suptitle('BSMI 測風塔 2016-2021 歷年颱風季風場與大氣壓力年際變化趨勢', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

trend_chart_path = os.path.join(OUTPUT_DIR, 'typhoon_season_trends.png')
plt.savefig(trend_chart_path, dpi=150, facecolor='#FAFBFC')
plt.close()
print(f"  [OK] 歷年颱風季趨勢統計圖已儲存至: {trend_chart_path}")

# =====================================================================
# 5. 輸出統計指標至 CSV
# =====================================================================
# 統計各年份 7-9 月的最大風速、平均風速、最低氣壓
summary_data = []
for yr in sorted(df_season_hourly['Year'].unique()):
    df_yr = df_season_hourly[df_season_hourly['Year'] == yr]
    summary_data.append({
        "年份": int(yr),
        "颱風季平均風速 (m/s)": df_yr['WS_100'].mean(),
        "颱風季最大風速 (m/s)": df_yr['WS_100'].max(),
        "最低大氣壓 (hPa)": df_yr['BP_93'].min(),
        "平均大氣壓 (hPa)": df_yr['BP_93'].mean()
    })

summary_df = pd.DataFrame(summary_data)
csv_path = os.path.join(OUTPUT_DIR, 'typhoon_annual_summary.csv')
summary_df.to_csv(csv_path, index=False)
print(f"  [OK] 颱風季歷年指標摘要表已儲存至: {csv_path}")
print("=" * 60 + "\n")
