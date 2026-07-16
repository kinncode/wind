#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================
  實驗九：局部微氣象突變事件預警系統
  Micro-Meteorological Sudden-Change Event Warning (LightGBM)

  資料: BSMI 測風塔 1-min 資料 (parquet, 2016-03 ~ 2021-10)
  任務: 預警未來 30 / 60 分鐘內是否有「微氣象突變事件」開始
        (鋒面 / 低壓槽 / 突發暴風雨 → 氣壓暴跌、濕度驟升、
         風速劇增、風向急轉)
  價值: 對照離岸風電運維指引 3.6.4 — 運維工作船 (SOV)
        海上吊裝作業與人員安全之提前預警

  方法論 (沿用實驗 04 框架):
    1. 複合訊號事件標籤 — 氣壓跌幅 / 濕度升幅 / 風速 ramp /
       風向轉變 四訊號，資料驅動百分位閾值，>=2 訊號共現
       + episode 合併，產出事件目錄 (event catalog)
    2. 多尺度時序特徵工程 (滾動統計 / 差分 / EWMA / 露點差 /
       圓形風向變率 / 時間編碼)
    3. Horizon-as-Feature (HaF): 30 / 60 分鐘提前量作為輸入特徵
    4. Expanding Window CV + 時間 embargo 防洩漏
    5. 事件級評估: POD / FAR / CSI / 平均提前量 / 每週誤報數
       並與 rule-based 基準 (氣壓速降規則) 對照

  執行方式:
    cd experiments/09_micro_meteo_warning
    python wind_event_warning.py
============================================================
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
    pkg_import = {"pandas": "pandas", "pyarrow": "pyarrow", "numpy": "numpy",
                  "scikit-learn": "sklearn", "lightgbm": "lightgbm",
                  "matplotlib": "matplotlib", "seaborn": "seaborn"}
    for pkg, mod in pkg_import.items():
        try:
            __import__(mod)
        except ImportError:
            print(f"  正在安裝必要套件 {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

install_packages()

import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, f1_score, precision_recall_curve)

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Microsoft JhengHei",
                                   "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────────────────
# Config — 所有可調參數集中於此
# ─────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..",
                                           "data", "BSMI_wind_1min_parquet"))
RESULT_DIR  = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ---- 事件標籤定義 (Label Definition) ----
SIGNAL_MODE     = "quad"   # "dual" = 氣壓+濕度 | "quad" = 氣壓+濕度+風速ramp+風向轉變
BP_DROP_PCTL    = 0.5      # BP_diff_60 低於此百分位 → 氣壓暴跌訊號 (%)
RH_RISE_PCTL    = 99.5     # RH_diff_60 高於此百分位 → 濕度驟升訊號 (%)
WS_RAMP_PCTL    = 99.5     # WS_diff_30 高於此百分位 → 風速劇增訊號 (%)
WD_SHIFT_DEG    = 45.0     # 30 分鐘內風向轉變超過此角度 → 風向急轉訊號 (度)
WD_MIN_WS       = 3.0      # 風速低於此值時風向無意義，不觸發風向訊號 (m/s)
MIN_SIGNALS     = 2        # 至少幾個訊號在共現窗內同時活躍才構成事件
CO_WINDOW       = 30       # 訊號共現窗 (分鐘)
MERGE_GAP       = 60       # 相鄰事件段間隔小於此值 → 合併為同一事件 (分鐘)
MIN_DURATION    = 10       # 事件最短持續時間，低於則視為雜訊剔除 (分鐘)
POST_EVENT_EXCL = 60       # 事件結束後排除取樣的緩衝期 (分鐘)

# ---- 時間序列參數 ----
WINDOW           = 180             # 特徵暖機長度 (分鐘)
WARNING_HORIZONS = [30, 60]        # HaF 預警提前量 (分鐘)
STRIDE           = 5               # 負樣本取樣步長 (分鐘); 正樣本一律全取
GAP_MIN          = 1.5             # 斷段閾值 (分鐘)
N_CV_SPLITS      = 5               # Expanding Window CV 折數
EMBARGO_MIN      = 240             # 訓練/測試間時間緩衝 (分鐘, >= WINDOW + max horizon)
VAL_TAIL_FRAC    = 0.10            # 每折訓練集尾端切出作 early stopping / 閾值選擇

# ---- LightGBM 參數 ----
LGBM_PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=10,
                   num_leaves=127, subsample=0.8, colsample_bytree=0.8,
                   min_child_samples=100, random_state=42, verbose=-1)
EARLY_STOP = 50

# ---- Rule-based 基準 ----
BASELINE_PCTL = 0.5   # 「BP_diff_30 低於訓練期此百分位 → 發警報」

print(f"SignalMode={SIGNAL_MODE}, MinSignals={MIN_SIGNALS}, CoWindow={CO_WINDOW}min")
print(f"Horizons={WARNING_HORIZONS}, Window={WINDOW}min, Embargo={EMBARGO_MIN}min")
print(f"CV_Splits={N_CV_SPLITS}, Stride={STRIDE}")

# ============================================================
# 1. 載入 Parquet 資料
# ============================================================
files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
print(f"\nFound {len(files)} parquet files")
if len(files) == 0:
    raise FileNotFoundError(
        f"在 {PARQUET_DIR} 中找不到任何 .parquet 檔案。\n"
        f"請將 Parquet 資料放在此目錄下再執行。")
df = pd.concat([pd.read_parquet(f) for f in files],
               ignore_index=False).sort_index()
df = df[~df.index.duplicated(keep="first")]
print(f"Total {len(df):,} rows, {df.index.min()} ~ {df.index.max()}")

# ============================================================
# 2. 基礎衍生量 (風速代表值 / 風向角 / 露點)
# ============================================================
df["WS_hub"] = (df["WS_100E"] + df["WS_100W"]) / 2   # 100m 輪轂風速代表值

def circular_shift_deg(sin_s, cos_s, k):
    """k 分鐘內風向轉變絕對角度 (度), 正確處理 0/360 環繞"""
    theta = np.arctan2(sin_s, cos_s)
    d = theta.diff(k)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(d.abs())

df["WD97_shift_30"] = circular_shift_deg(df["WD_97_sin"], df["WD_97_cos"], 30)
df["WD97_shift_60"] = circular_shift_deg(df["WD_97_sin"], df["WD_97_cos"], 60)

# 露點與溫度露點差 (T - Td, 越小越接近飽和 → 降雨前兆)
_RH = np.clip(df["RH_95"], 1, 100)
_g  = (17.27 * df["AT_95"]) / (237.7 + df["AT_95"]) + np.log(_RH / 100)
df["Td_95"]     = 237.7 * _g / (17.27 - _g)
df["T_Td_diff"] = df["AT_95"] - df["Td_95"]

# 標籤定義所需的核心差分
df["BP_diff_60"] = df["BP_93"].diff(60)
df["RH_diff_60"] = df["RH_95"].diff(60)
df["WS_diff_30"] = df["WS_hub"].diff(30)

# ============================================================
# 3. 複合訊號事件標籤 (Event Catalog)
# ============================================================
# 註: 事件「定義」使用全期間氣候百分位屬離線目錄建構, 為定義選擇
#     而非特徵洩漏; 模型輸入特徵仍嚴格只用過去資訊。
def build_signals(df):
    thr = {
        "bp_drop": np.nanpercentile(df["BP_diff_60"], BP_DROP_PCTL),
        "rh_rise": np.nanpercentile(df["RH_diff_60"], RH_RISE_PCTL),
        "ws_ramp": np.nanpercentile(df["WS_diff_30"], WS_RAMP_PCTL),
    }
    sig = pd.DataFrame(index=df.index)
    sig["sig_bp"] = df["BP_diff_60"] <= thr["bp_drop"]
    sig["sig_rh"] = df["RH_diff_60"] >= thr["rh_rise"]
    if SIGNAL_MODE == "quad":
        sig["sig_ws"] = df["WS_diff_30"] >= thr["ws_ramp"]
        sig["sig_wd"] = (df["WD97_shift_30"] >= WD_SHIFT_DEG) & \
                        (df["WS_hub"] >= WD_MIN_WS)
    print("\n=== 事件訊號閾值 (資料驅動) ===")
    print(f"  氣壓 60min 跌幅  <= {thr['bp_drop']:+.3f} hPa "
          f"(第 {BP_DROP_PCTL} 百分位)")
    print(f"  濕度 60min 升幅  >= {thr['rh_rise']:+.3f} %RH "
          f"(第 {RH_RISE_PCTL} 百分位)")
    if SIGNAL_MODE == "quad":
        print(f"  風速 30min 躍升  >= {thr['ws_ramp']:+.3f} m/s "
              f"(第 {WS_RAMP_PCTL} 百分位)")
        print(f"  風向 30min 轉變  >= {WD_SHIFT_DEG:.0f} 度 "
              f"(且 WS_hub >= {WD_MIN_WS} m/s)")
    for c in sig.columns:
        print(f"  {c}: 觸發率 {sig[c].mean()*100:.3f}%")
    return sig, thr


def build_episodes(sig):
    """訊號共現 → 事件段 (episode) 合併與過濾, 回傳事件目錄"""
    # 各訊號在共現窗內任一時刻觸發即視為「活躍」
    active = sig.rolling(CO_WINDOW, min_periods=1).max()
    composite = (active.sum(axis=1) >= MIN_SIGNALS)

    # 連續正段落 → episodes
    grp = (composite != composite.shift()).cumsum()
    episodes = []
    for _, seg in composite.groupby(grp):
        if not seg.iloc[0]:
            continue
        episodes.append([seg.index[0], seg.index[-1]])

    # 合併間隔 < MERGE_GAP 的相鄰事件
    merged = []
    for ep in episodes:
        if merged and (ep[0] - merged[-1][1]) <= pd.Timedelta(minutes=MERGE_GAP):
            merged[-1][1] = ep[1]
        else:
            merged.append(ep)

    # 剔除過短事件
    merged = [ep for ep in merged
              if (ep[1] - ep[0]) >= pd.Timedelta(minutes=MIN_DURATION)]

    cat = pd.DataFrame(merged, columns=["start", "end"])
    cat["duration_min"] = (cat["end"] - cat["start"]).dt.total_seconds() / 60
    return cat


signals, sig_thr = build_signals(df)
catalog = build_episodes(signals)

# 事件期間各訊號是否曾觸發 + 事件內最大氣壓跌幅 (供目錄檢視 / 颱風比對)
for c in signals.columns:
    catalog[c] = [bool(signals.loc[s:e, c].any())
                  for s, e in zip(catalog["start"], catalog["end"])]
catalog["max_bp_drop_60"] = [df.loc[s:e, "BP_diff_60"].min()
                             for s, e in zip(catalog["start"], catalog["end"])]
catalog["max_ws_hub"] = [df.loc[s:e, "WS_hub"].max()
                         for s, e in zip(catalog["start"], catalog["end"])]

print(f"\n=== 事件目錄 ===")
print(f"  事件總數: {len(catalog)}")
print(f"  平均持續: {catalog['duration_min'].mean():.1f} min, "
      f"中位數: {catalog['duration_min'].median():.1f} min")
print(f"  事件時間占比: "
      f"{catalog['duration_min'].sum() / (len(df)) * 100:.2f}%")
print("\n  [人工校驗建議] 以下為氣壓跌幅最劇烈的 10 場事件, "
      "可對照 2016–2021 侵台颱風與鋒面紀錄:")
top10 = catalog.nsmallest(10, "max_bp_drop_60")
for _, r in top10.iterrows():
    print(f"    {r['start']}  ~  {r['end']}  "
          f"(ΔBP={r['max_bp_drop_60']:+.2f} hPa, "
          f"WSmax={r['max_ws_hub']:.1f} m/s)")

catalog.to_csv(os.path.join(RESULT_DIR, "event_catalog.csv"), index=False)

# 事件季節/年度分布圖
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
catalog["start"].dt.month.value_counts().sort_index().reindex(
    range(1, 13), fill_value=0).plot(
    kind="bar", ax=axes[0], color="#2b7a9e")
axes[0].set_title("事件月份分布 (季風/梅雨/颱風季節性)")
axes[0].set_xlabel("月份"); axes[0].set_ylabel("事件數")
catalog["start"].dt.year.value_counts().sort_index().plot(
    kind="bar", ax=axes[1], color="#9e5a2b")
axes[1].set_title("事件年度分布")
axes[1].set_xlabel("年份"); axes[1].set_ylabel("事件數")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "event_seasonal_distribution.png"), dpi=120)
plt.close()

# ============================================================
# 4. 時序特徵工程 (嚴格因果: 只用過去資訊)
# ============================================================
def add_ts_features(df):
    feat = pd.DataFrame(index=df.index)

    # ---- 氣壓 (最核心的前兆變數) ----
    feat["BP"] = df["BP_93"]
    for k in [10, 30, 60, 180]:            # 180min ≈ 天氣學標準 3 小時氣壓趨勢
        feat[f"BP_diff_{k}"] = df["BP_93"].diff(k)
    for w in [30, 60, 180]:
        feat[f"BP_std_{w}"] = df["BP_93"].rolling(w, min_periods=w//2).std()
    feat["BP_ewm_30"] = df["BP_93"].ewm(span=30).mean() - df["BP_93"]

    # ---- 濕度 / 溫度 / 露點差 ----
    feat["RH"] = df["RH_95"]
    for k in [30, 60, 180]:
        feat[f"RH_diff_{k}"] = df["RH_95"].diff(k)
    feat["RH_std_60"] = df["RH_95"].rolling(60, min_periods=30).std()
    feat["AT"] = df["AT_95"]
    for k in [30, 60, 180]:
        feat[f"AT_diff_{k}"] = df["AT_95"].diff(k)
    feat["T_Td"] = df["T_Td_diff"]
    feat["T_Td_diff_60"] = df["T_Td_diff"].diff(60)

    # ---- 風速 (ramp 前兆) ----
    feat["WS_hub"] = df["WS_hub"]
    feat["WS_38W"] = df["WS_38W"]
    for k in [10, 30, 60]:
        feat[f"WS_diff_{k}"] = df["WS_hub"].diff(k)
    for w in [10, 30, 60]:
        feat[f"WS_std_{w}"] = df["WS_hub"].rolling(w, min_periods=w//2).std()
    feat["WS_max_60"] = df["WS_hub"].rolling(60, min_periods=30).max()
    feat["TI_hub_30"] = (df["WS_hub"].rolling(30, min_periods=10).std()
                         / df["WS_hub"].clip(lower=0.5))
    # 風切指數 (大氣穩定度共變量)
    feat["alpha_38_100"] = (np.log(df["WS_hub"].clip(0.3)
                                   / df["WS_38W"].clip(0.3))
                            / np.log(100.0 / 38.0))
    feat["alpha_roll_30"] = feat["alpha_38_100"].rolling(
        30, min_periods=15).mean()

    # ---- 風向 (圓形變率 + 穩定度 r 值) ----
    feat["WD_97_sin"] = df["WD_97_sin"]
    feat["WD_97_cos"] = df["WD_97_cos"]
    feat["WD97_shift_30"] = df["WD97_shift_30"]
    feat["WD97_shift_60"] = df["WD97_shift_60"]
    _w = 30
    feat["WD_97_r"] = np.sqrt(
        df["WD_97_sin"].rolling(_w, min_periods=10).mean() ** 2 +
        df["WD_97_cos"].rolling(_w, min_periods=10).mean() ** 2)

    # ---- 時間週期編碼 (季風季節 / 日夜對流) ----
    hr  = df.index.hour + df.index.minute / 60
    mon = df.index.month
    feat["hour_sin"]  = np.sin(2 * np.pi * hr / 24)
    feat["hour_cos"]  = np.cos(2 * np.pi * hr / 24)
    feat["month_sin"] = np.sin(2 * np.pi * mon / 12)
    feat["month_cos"] = np.cos(2 * np.pi * mon / 12)
    return feat


feat = add_ts_features(df)
FEATURE_COLS = list(feat.columns) + ["horizon"]
print(f"\n特徵維度: {len(FEATURE_COLS)} (含 HaF horizon)")

# ============================================================
# 5. 建立 HaF 樣本 (排除事件中時段, 防止「已在暴風中」的假預警)
# ============================================================
def find_segments(index, gap_min=GAP_MIN):
    """依時間斷點切出連續段落 (回傳整數位置區間)"""
    gaps = np.where(np.diff(index.values).astype("timedelta64[s]")
                    > np.timedelta64(int(gap_min * 60), "s"))[0]
    starts = np.concatenate([[0], gaps + 1])
    ends   = np.concatenate([gaps, [len(index) - 1]])
    return list(zip(starts, ends))


def build_exclusion_mask(index, catalog):
    """事件進行中 + 事件後緩衝期 → 不作為訓練/評估樣本"""
    mask = np.zeros(len(index), dtype=bool)
    for s, e in zip(catalog["start"], catalog["end"]):
        e_buf = e + pd.Timedelta(minutes=POST_EVENT_EXCL)
        mask |= (index >= s) & (index <= e_buf)
    return mask


def build_haf_samples(index, feat, catalog, segments,
                      horizons=WARNING_HORIZONS, stride=STRIDE):
    """
    每個取樣時刻 t × 每個 horizon h → 一筆樣本
    label = 1 若有任一事件「開始」落於 (t, t+h]
    正樣本步長 1 全取, 負樣本依 stride 取樣 (控制資料量)
    """
    event_starts = catalog["start"].values.astype("datetime64[ns]")
    excl = build_exclusion_mask(index, catalog)
    feat_ok = ~feat.isna().any(axis=1).values

    X_list, y_list, t_list, h_list = [], [], [], []
    fvals = feat.values
    idx64 = index.values.astype("datetime64[ns]")
    max_h = max(horizons)

    for seg_s, seg_e in segments:
        lo = seg_s + WINDOW
        hi = seg_e - max_h
        if hi <= lo:
            continue
        for pos in range(lo, hi + 1):
            if excl[pos] or not feat_ok[pos]:
                continue
            t = idx64[pos]
            labels = {}
            for h in horizons:
                # 事件開始時間落於 (t, t+h] ?
                left  = np.searchsorted(event_starts, t, side="right")
                right = np.searchsorted(event_starts,
                                        t + np.timedelta64(h, "m"),
                                        side="right")
                labels[h] = int(right > left)
            is_pos = any(labels.values())
            if not is_pos and (pos % stride != 0):
                continue
            for h in horizons:
                X_list.append(fvals[pos])
                y_list.append(labels[h])
                t_list.append(t)
                h_list.append(h)

    X = np.column_stack([np.asarray(X_list, dtype=np.float32),
                         np.asarray(h_list, dtype=np.float32)])
    y = np.asarray(y_list, dtype=np.int8)
    ts = np.asarray(t_list)
    order = np.argsort(ts, kind="stable")   # 依基準時間排序 → 時序 CV
    return X[order], y[order], ts[order]


segments = find_segments(df.index)
print(f"連續段落數: {len(segments)}")
X, y, ts = build_haf_samples(df.index, feat, catalog, segments)
print(f"樣本總數: {len(y):,}  正樣本: {y.sum():,} ({y.mean()*100:.3f}%)")

# ============================================================
# 6. Expanding Window CV + 時間 Embargo
# ============================================================
def choose_threshold(y_true, y_prob):
    """在驗證集上選 F1 最大化的機率閾值"""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best = np.argmax(f1[:-1]) if len(thr) > 0 else 0
    return float(thr[best]) if len(thr) > 0 else 0.5


def train_expanding_cv(X, y, ts):
    n = len(y)
    block = n // (N_CV_SPLITS + 1)
    fold_metrics, models = [], []
    last_fold = None

    print("\n" + "=" * 60)
    print(f" Expanding Window CV ({N_CV_SPLITS} folds, "
          f"embargo={EMBARGO_MIN} min)")
    print("=" * 60)

    for fold in range(N_CV_SPLITS):
        train_end = (fold + 1) * block
        test_end  = min((fold + 2) * block, n)

        # 時間 embargo: 測試樣本基準時間須晚於訓練尾端 + EMBARGO_MIN
        cutoff = ts[train_end - 1] + np.timedelta64(EMBARGO_MIN, "m")
        test_start = train_end + np.searchsorted(ts[train_end:test_end],
                                                 cutoff, side="right")
        if test_start >= test_end:
            print(f"  Fold {fold+1}: skipped (embargo too large)")
            continue

        # 訓練尾端切出驗證段 (early stopping + 閾值選擇)
        val_n = max(int(train_end * VAL_TAIL_FRAC), 1000)
        tr_idx = np.arange(0, train_end - val_n)
        va_idx = np.arange(train_end - val_n, train_end)
        te_idx = np.arange(test_start, test_end)

        pos_tr = y[tr_idx].sum()
        if pos_tr < 10 or y[te_idx].sum() < 1:
            print(f"  Fold {fold+1}: skipped (正樣本不足 "
                  f"train={pos_tr}, test={y[te_idx].sum()})")
            continue
        spw = (len(tr_idx) - pos_tr) / max(pos_tr, 1)

        clf = LGBMClassifier(**LGBM_PARAMS, objective="binary",
                             scale_pos_weight=spw)
        clf.fit(X[tr_idx], y[tr_idx],
                eval_set=[(X[va_idx], y[va_idx])],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                           lgb.log_evaluation(0)])

        thr_f1 = choose_threshold(y[va_idx],
                                  clf.predict_proba(X[va_idx])[:, 1])
        prob_te = clf.predict_proba(X[te_idx])[:, 1]
        pred_te = (prob_te >= thr_f1).astype(int)

        m = dict(fold=fold + 1,
                 n_train=len(tr_idx), n_test=len(te_idx),
                 pos_rate_test=float(y[te_idx].mean()),
                 threshold=thr_f1,
                 pr_auc=average_precision_score(y[te_idx], prob_te),
                 precision=precision_score(y[te_idx], pred_te,
                                           zero_division=0),
                 recall=recall_score(y[te_idx], pred_te, zero_division=0),
                 f1=f1_score(y[te_idx], pred_te, zero_division=0))
        fold_metrics.append(m)
        models.append(clf)
        last_fold = dict(clf=clf, thr=thr_f1, te_idx=te_idx,
                         tr_end_time=ts[train_end - 1])
        print(f"  Fold {fold+1}/{N_CV_SPLITS}: "
              f"train={len(tr_idx):,} test={len(te_idx):,} "
              f"pos={y[te_idx].mean()*100:.3f}% | "
              f"PR-AUC={m['pr_auc']:.4f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} F1={m['f1']:.3f}")

    return models, fold_metrics, last_fold


models, fold_metrics, last_fold = train_expanding_cv(X, y, ts)
report = pd.DataFrame(fold_metrics)
report.loc["mean"] = report.mean(numeric_only=True)
report.to_csv(os.path.join(RESULT_DIR, "cv_evaluation_report.csv"),
              index=False)
print("\n=== 逐點 (sample-level) CV 平均 ===")
print(report.tail(1)[["pr_auc", "precision", "recall", "f1"]]
      .round(4).to_string(index=False))

# 最後一折 PR 曲線
if last_fold is not None:
    clf, te_idx = last_fold["clf"], last_fold["te_idx"]
    plt.figure(figsize=(6, 5))
    for h in WARNING_HORIZONS:
        h_mask = X[te_idx, -1] == h
        if y[te_idx][h_mask].sum() == 0:
            continue
        prob = clf.predict_proba(X[te_idx][h_mask])[:, 1]
        prec, rec, _ = precision_recall_curve(y[te_idx][h_mask], prob)
        ap = average_precision_score(y[te_idx][h_mask], prob)
        plt.plot(rec, prec, label=f"h={h}min (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR 曲線 (最後一折測試集, 分 horizon)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "pr_curves.png"), dpi=120)
    plt.close()

# ============================================================
# 7. 事件級評估 (POD / FAR / CSI / 提前量) — 這才是運維要看的
# ============================================================
def consolidate_runs(times, max_gap_min=15):
    """把離散警報分鐘合併為警報段 [(start, end), ...]"""
    if len(times) == 0:
        return []
    runs, s, prev = [], times[0], times[0]
    for t in times[1:]:
        if (t - prev) > np.timedelta64(max_gap_min, "m"):
            runs.append((s, prev)); s = t
        prev = t
    runs.append((s, prev))
    return runs


def event_level_eval(warn_times, event_starts, horizon,
                     test_span_days, in_event_fn):
    """
    warn_times   : 發出警報的時間點 array (已排序, 不含事件中時段)
    event_starts : 測試期內事件開始時間 array
    hit  : 事件開始前 [start-h, start) 內曾發警報
    FA   : 警報段起點後 h 分鐘內 (至段尾+h) 無任何事件開始
    """
    hits, leads = 0, []
    for s in event_starts:
        w = warn_times[(warn_times >= s - np.timedelta64(horizon, "m"))
                       & (warn_times < s)]
        if len(w) > 0:
            hits += 1
            leads.append((s - w[0]) / np.timedelta64(1, "m"))
    misses = len(event_starts) - hits

    fa = 0
    for rs, re in consolidate_runs(warn_times):
        lo = np.searchsorted(event_starts, rs, side="left")
        hi = np.searchsorted(event_starts,
                             re + np.timedelta64(horizon, "m"), side="right")
        if hi <= lo:
            fa += 1

    pod = hits / max(hits + misses, 1)
    far = fa / max(fa + hits, 1)
    csi = hits / max(hits + misses + fa, 1)
    return dict(n_events=len(event_starts), hits=hits, misses=misses,
                false_alarms=fa,
                POD=round(pod, 4), FAR=round(far, 4), CSI=round(csi, 4),
                FA_per_week=round(fa / max(test_span_days / 7, 1e-9), 3),
                mean_lead_min=round(float(np.mean(leads)), 1) if leads else np.nan)


ev_rows = []
if last_fold is not None:
    clf, thr_op = last_fold["clf"], last_fold["thr"]
    test_t0 = ts[last_fold["te_idx"]][0]
    test_t1 = ts[last_fold["te_idx"]][-1]
    span_days = (test_t1 - test_t0) / np.timedelta64(1, "D")
    print(f"\n=== 事件級評估 (測試期 {test_t0} ~ {test_t1}, "
          f"{span_days:.0f} 天) ===")

    # 測試期逐分鐘掃描 (排除事件中/緩衝期, 模擬線上連續運行)
    excl_full = build_exclusion_mask(df.index, catalog)
    feat_ok   = ~feat.isna().any(axis=1).values
    scan_mask = ((df.index.values >= test_t0) & (df.index.values <= test_t1)
                 & ~excl_full & feat_ok)
    scan_pos  = np.where(scan_mask)[0]
    scan_time = df.index.values[scan_pos]
    Xscan_base = feat.values[scan_pos].astype(np.float32)

    ev_starts_all = catalog["start"].values.astype("datetime64[ns]")
    ev_test = ev_starts_all[(ev_starts_all >= test_t0)
                            & (ev_starts_all <= test_t1)]
    print(f"  測試期事件數: {len(ev_test)}, 操作閾值: {thr_op:.3f}")

    # Rule-based 基準閾值: 由「訓練期」BP_diff_30 百分位決定
    tr_mask = df.index.values <= last_fold["tr_end_time"]
    rule_thr = np.nanpercentile(feat.loc[tr_mask, "BP_diff_30"],
                                BASELINE_PCTL)
    print(f"  基準規則: BP_diff_30 <= {rule_thr:+.3f} hPa → 警報")

    for h in WARNING_HORIZONS:
        Xscan = np.column_stack([Xscan_base,
                                 np.full(len(scan_pos), h, np.float32)])
        prob = clf.predict_proba(Xscan)[:, 1]
        warn_t = scan_time[prob >= thr_op]
        m = event_level_eval(warn_t, ev_test, h, span_days, None)
        m.update(model="LightGBM", horizon=h)
        ev_rows.append(m)

        rule_warn = scan_time[Xscan_base[:, FEATURE_COLS.index("BP_diff_30")]
                              <= rule_thr]
        mb = event_level_eval(rule_warn, ev_test, h, span_days, None)
        mb.update(model="Rule (BP速降)", horizon=h)
        ev_rows.append(mb)

    ev_report = pd.DataFrame(ev_rows)[
        ["model", "horizon", "n_events", "hits", "misses", "false_alarms",
         "POD", "FAR", "CSI", "FA_per_week", "mean_lead_min"]]
    ev_report.to_csv(os.path.join(RESULT_DIR, "event_level_report.csv"),
                     index=False)
    print("\n" + ev_report.to_string(index=False))

# ============================================================
# 8. 特徵重要性 + 預警時間軸範例圖
# ============================================================
if last_fold is not None:
    clf = last_fold["clf"]
    imp = pd.Series(clf.booster_.feature_importance("gain"),
                    index=FEATURE_COLS).sort_values()[-25:]
    plt.figure(figsize=(8, 8))
    imp.plot(kind="barh", color="#2b7a9e")
    plt.title("LightGBM 特徵重要性 (gain, Top 25)\n"
              "— 哪個前兆對突變預警最有預測力?")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "feature_importance.png"), dpi=120)
    plt.close()

    # 預警時間軸: 取測試期氣壓跌幅最劇烈的事件, 畫前後 12 小時
    cat_test = catalog[(catalog["start"] >= pd.Timestamp(test_t0))
                       & (catalog["start"] <= pd.Timestamp(test_t1))]
    if len(cat_test) > 0:
        ev = cat_test.nsmallest(1, "max_bp_drop_60").iloc[0]
        w0 = ev["start"] - pd.Timedelta(hours=12)
        w1 = ev["end"]   + pd.Timedelta(hours=6)
        win = df.loc[w0:w1]
        wpos = np.where((scan_time >= np.datetime64(w0))
                        & (scan_time <= np.datetime64(w1)))[0]
        Xw = np.column_stack([Xscan_base[wpos],
                              np.full(len(wpos), WARNING_HORIZONS[-1],
                                      np.float32)])
        pw = clf.predict_proba(Xw)[:, 1] if len(wpos) else np.array([])

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(win.index, win["BP_93"], color="#333")
        axes[0].set_ylabel("氣壓 (hPa)")
        axes[1].plot(win.index, win["RH_95"], color="#2b7a9e")
        axes[1].set_ylabel("濕度 (%)")
        axes[2].plot(win.index, win["WS_hub"], color="#9e5a2b")
        axes[2].set_ylabel("WS_hub (m/s)")
        if len(wpos):
            axes[3].plot(pd.to_datetime(scan_time[wpos]), pw,
                         color="#c0392b")
        axes[3].axhline(last_fold["thr"], ls="--", color="gray",
                        label=f"操作閾值 {last_fold['thr']:.2f}")
        axes[3].set_ylabel(f"預警機率\n(h={WARNING_HORIZONS[-1]}min)")
        axes[3].set_ylim(0, 1); axes[3].legend(loc="upper left")
        for ax in axes:
            ax.axvspan(ev["start"], ev["end"], color="red", alpha=0.12)
            ax.grid(alpha=0.3)
        axes[0].set_title(f"預警時間軸範例 — 事件 {ev['start']} "
                          f"(紅色區塊, ΔBP={ev['max_bp_drop_60']:+.1f} hPa)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "warning_timeline.png"), dpi=120)
        plt.close()

print(f"\n[OK] 全部完成, 結果輸出於: {RESULT_DIR}")
print("   - event_catalog.csv              事件目錄 (含颱風比對欄位)")
print("   - event_seasonal_distribution.png 事件季節/年度分布")
print("   - cv_evaluation_report.csv        逐點 CV 指標 (PR-AUC 等)")
print("   - pr_curves.png                   分 horizon PR 曲線")
print("   - event_level_report.csv          事件級 POD/FAR/CSI vs 基準")
print("   - feature_importance.png          前兆特徵重要性")
print("   - warning_timeline.png            預警時間軸範例")
