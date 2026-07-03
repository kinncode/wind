#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BSMI 大氣穩定度預測系統
Atmospheric Stability Prediction via Wind Profile Power Law

核心優勢：利用本測站 4 個高度的風速計（38m / 69m / 100mE / 100mW），
計算風速剖面冪律指數 α，量化大氣穩定度並預測其未來變化。

> 這是一般測站（只有 1–2 高度）無法複製的獨特能力。

本版本已修改為可在本地環境執行（無需 Google Colab / Google Drive）。
"""

# ============================================================
# Cell 1: 安裝相依套件（本地版本）
# ============================================================
import subprocess, sys

def install_packages():
    """安裝必要套件（若尚未安裝）"""
    packages = [
        'xgboost', 'lightgbm', 'pyarrow', 'pandas',
        'scikit-learn', 'joblib', 'matplotlib', 'seaborn'
    ]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f'Installing {pkg}...')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

install_packages()

# ============================================================
# Cell 2: (已移除 Google Colab drive.mount — 本地不需要)
# ============================================================

# ============================================================
# Cell 3: 匯入套件 & 設定參數
# ============================================================
import numpy as np, pandas as pd, math, warnings, glob, os, joblib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC','Microsoft JhengHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---- 本地路徑設定 ----
# 以腳本所在目錄為基準，自動定位資料與模型資料夾
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = os.path.join(SCRIPT_DIR, 'BSMI_wind_1min_parquet')
MODEL_DIR   = os.path.join(SCRIPT_DIR, 'models_stability')
os.makedirs(MODEL_DIR, exist_ok=True)

HEIGHTS = np.array([38.0, 69.0, 100.0])
LN_HEIGHTS = np.log(HEIGHTS)
STABLE_THRESHOLD = 0.20   # binary: alpha >= 0.20 → stable, else not-stable
WINDOW=60; HORIZONS=[15,30,60,90,120]; STRIDE=10; GAP_MIN=1.5
VAL_RATIO=0.10; TEST_RATIO=0.10
LGBM_PARAMS = dict(n_estimators=1000, learning_rate=0.08, max_depth=14,
                    num_leaves=256, subsample=0.8, colsample_bytree=0.8,
                    min_child_samples=50, random_state=42, verbose=-1)
EARLY_STOP = 50
print(f'Window={WINDOW}, Horizons={HORIZONS}, Stride={STRIDE}')

# ============================================================
# Cell 4: 載入 Parquet 資料
# ============================================================
files = sorted(glob.glob(os.path.join(PARQUET_DIR, '*.parquet')))
print(f'Found {len(files)} parquet files')
if len(files) == 0:
    raise FileNotFoundError(
        f'在 {PARQUET_DIR} 中找不到任何 .parquet 檔案。\n'
        f'請將 Parquet 資料放在此目錄下再執行。'
    )
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=False).sort_index()
print(f'Total {len(df):,} rows, {df.index.min()} ~ {df.index.max()}')

# ============================================================
# Cell 5: 特徵工程 — 計算 alpha 與穩定度特徵
# ============================================================
def compute_alpha(df):
    ws100 = (df['WS_100E']+df['WS_100W'])/2
    ws69, ws38 = df['WS_69W'], df['WS_38W']
    WS_mat = np.column_stack([ws38.values, ws69.values, ws100.values])
    valid = (WS_mat > 0.1).all(axis=1)
    LN_WS = np.log(np.clip(WS_mat, 0.1, None))
    lnz_mean = LN_HEIGHTS.mean(); lnws_mean = LN_WS.mean(axis=1)
    numer = np.zeros(len(df)); denom = 0.0
    for i in range(3):
        numer += (LN_HEIGHTS[i]-lnz_mean)*(LN_WS[:,i]-lnws_mean)
        denom += (LN_HEIGHTS[i]-lnz_mean)**2
    alpha = numer/denom; alpha[~valid]=np.nan
    return alpha

def add_stability_features(df):
    df = df.copy()
    ws100 = (df['WS_100E']+df['WS_100W'])/2; ws69=df['WS_69W']; ws38=df['WS_38W']
    df['shear_low']=ws69-ws38; df['shear_high']=ws100-ws69
    df['shear_ratio']=df['shear_high']/(df['shear_low'].abs()+0.01)
    df['shear_total']=ws100-ws38
    df['alpha']=compute_alpha(df)
    df['alpha_30']=df['alpha'].rolling(30,min_periods=15).mean()
    df['alpha_std_30']=df['alpha'].rolling(30,min_periods=15).std()
    m100=ws100.rolling(10,min_periods=5).mean(); m38=ws38.rolling(10,min_periods=5).mean()
    s100=ws100.rolling(10,min_periods=5).std(); s38=ws38.rolling(10,min_periods=5).std()
    df['TI_100']=s100/(m100+0.01); df['TI_38']=s38/(m38+0.01)
    df['TI_ratio']=df['TI_100']/(df['TI_38']+0.01)
    df['BP_diff_30']=df['BP_93']-df['BP_93'].shift(30)
    df['BP_diff_60']=df['BP_93']-df['BP_93'].shift(60)
    df['BP_std_60']=df['BP_93'].rolling(60,min_periods=30).std()
    df['AT_diff_60']=df['AT_95']-df['AT_95'].shift(60)
    df['RH_diff_60']=df['RH_95']-df['RH_95'].shift(60)
    RH=np.clip(df['RH_95'],1,100)
    g=(17.27*df['AT_95'])/(237.7+df['AT_95'])+np.log(RH/100)
    df['Td_95']=(237.7*g)/(17.27-g); df['T_Td_diff']=df['AT_95']-df['Td_95']
    df['WS_shear']=ws100-ws38; df['WS_mean']=(df['WS_100E']+df['WS_100W']+ws69+ws38)/4
    df['WS_std_30']=ws100.rolling(30,min_periods=15).std()
    df['hour_sin']=np.sin(2*math.pi*df.index.hour/24); df['hour_cos']=np.cos(2*math.pi*df.index.hour/24)
    df['month_sin']=np.sin(2*math.pi*df.index.month/12); df['month_cos']=np.cos(2*math.pi*df.index.month/12)
    return df

df = add_stability_features(df)
BASE_FEATS=['WS_100E','WS_100W','WS_69W','WS_38W','WD_97_sin','WD_97_cos','WD_35_sin','WD_35_cos','AT_95','RH_95','BP_93']
STABILITY_FEATS=['shear_low','shear_high','shear_ratio','shear_total','alpha','alpha_30','alpha_std_30','TI_100','TI_38','TI_ratio']
WEATHER_FEATS=['BP_diff_30','BP_diff_60','BP_std_60','AT_diff_60','RH_diff_60','T_Td_diff','WS_shear','WS_mean','WS_std_30']
TIME_FEATS=['hour_sin','hour_cos','month_sin','month_cos']
FEATURES=BASE_FEATS+STABILITY_FEATS+WEATHER_FEATS
df=df.dropna(subset=FEATURES+TIME_FEATS)
print(f'After features: {len(df):,} rows, {len(FEATURES)+len(TIME_FEATS)} features/step')
print(f'  Base:{len(BASE_FEATS)} Stability:{len(STABILITY_FEATS)} Weather:{len(WEATHER_FEATS)} Time:{len(TIME_FEATS)}')

# ============================================================
# Cell 6: 繪製穩定度分佈圖
# ============================================================
fig, axes = plt.subplots(1,3,figsize=(16,4.5))
a=df['alpha'].dropna()
ax=axes[0]; ax.hist(a.clip(-1,1.5),bins=80,color='steelblue',alpha=0.8,edgecolor='white')
for thr,col,lbl in [(0.10,'orange','unstable/neutral'),(0.20,'red','neutral/stable')]:
    ax.axvline(thr,color=col,ls='--',lw=2,label=f'a={thr}')
ax.set_xlabel('alpha'); ax.set_ylabel('Count'); ax.set_title('Wind Profile Power Law Exponent alpha'); ax.legend(fontsize=8)
ax=axes[1]; hourly=df.groupby(df.index.hour)['alpha'].agg(['mean','std'])
ax.fill_between(hourly.index,hourly['mean']-hourly['std'],hourly['mean']+hourly['std'],alpha=0.2,color='steelblue')
ax.plot(hourly.index,hourly['mean'],'o-',color='steelblue',lw=2)
ax.axhline(0.10,color='orange',ls=':',lw=1); ax.axhline(0.20,color='red',ls=':',lw=1)
ax.set_xlabel('Hour'); ax.set_ylabel('alpha'); ax.set_title('Diurnal Pattern (night>day = correct)'); ax.set_xticks(range(0,24,3))
ax=axes[2]; n_stable=(a>=STABLE_THRESHOLD).sum(); n_notstable=(a<STABLE_THRESHOLD).sum()
ax.pie([n_notstable, n_stable],labels=['Not-Stable\n(\u03b1<0.20)','Stable\n(\u03b1\u22650.20)'],colors=['#5BA3E6','#FF8C42'],autopct='%1.1f%%',startangle=90)
ax.set_title('Stability Class Distribution')
plt.tight_layout(); plt.savefig(os.path.join(MODEL_DIR,'stability_distribution.png'),dpi=120); plt.show()
night=df.loc[df.index.hour.isin([20,21,22,23,0,1,2,3,4]),'alpha'].mean()
day=df.loc[df.index.hour.isin([10,11,12,13,14,15,16]),'alpha'].mean()
print(f'Night alpha={night:.3f}  Day alpha={day:.3f}  diff={night-day:.3f}  (night>day = physics correct)')

# ============================================================
# Cell 7: 建構穩定度樣本
# ============================================================
def build_stability_samples(df, horizon, window=WINDOW, stride=STRIDE):
    feat_cols=FEATURES+TIME_FEATS; arr=df[feat_cols].values.astype(np.float32)
    F=len(feat_cols) # Define F
    alpha_arr=df['alpha_30'].values.astype(np.float32)  # smoothed target
    idx=df.index # Define idx properly
    gaps=np.where(np.diff(idx.asi8)>int(GAP_MIN*60*1e9))[0]+1
    segs=np.split(np.arange(len(df)),gaps); total=0; vs=[]
    for s in segs:
        ms=len(s)-window-horizon
        if ms<=0: continue
        total+=(ms+stride-1)//stride; vs.append(s)
    X=np.empty((total,window*F),dtype=np.float32); y_r=np.empty(total,dtype=np.float32); k=0
    for s in vs:
        ms=len(s)-window-horizon
        for i in range(0,ms,stride):
            X[k]=arr[s[i:i+window]].ravel(); y_r[k]=alpha_arr[s[i+window+horizon-1]]; k+=1
    X,y_r=X[:k],y_r[:k]
    y_c=(y_r >= STABLE_THRESHOLD).astype(np.int32)  # binary: 0=not-stable, 1=stable
    v=np.isfinite(y_r); return X[v],y_r[v],y_c[v]

samples={}
for h in HORIZONS:
    X,yr,yc=build_stability_samples(df,h); n=len(X)
    t=int(n*(1-TEST_RATIO)); v=int(n*(1-TEST_RATIO-VAL_RATIO))
    samples[h]=dict(X_tr=X[:v],y_reg_tr=yr[:v],y_cls_tr=yc[:v],
                     X_va=X[v:t],y_reg_va=yr[v:t],y_cls_va=yc[v:t],
                     X_te=X[t:],y_reg_te=yr[t:],y_cls_te=yc[t:])
    u,c=np.unique(yc[t:],return_counts=True)
    cn={0:'Not-Stable',1:'Stable'}
    print(f't+{h:>3}min  n={n:,}  train={v:,}  val={t-v:,}  test={n-t:,}  [{" ".join(f"{cn[ui]}={ci}" for ui,ci in zip(u,c))}]')

# ============================================================
# Cell 8: 定義訓練 & 評估函式
# ============================================================
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, f1_score, classification_report)
import lightgbm as lgb

def train_stability_models(samples, horizon):
    sp=samples[horizon]
    reg=LGBMRegressor(**LGBM_PARAMS)
    reg.fit(sp['X_tr'],sp['y_reg_tr'],eval_set=[(sp['X_va'],sp['y_reg_va'])],
            callbacks=[lgb.early_stopping(EARLY_STOP,verbose=False),lgb.log_evaluation(0)])
    print(f'  Reg best_iter={reg.best_iteration_}')
    clf=LGBMClassifier(**LGBM_PARAMS,objective='binary')
    clf.fit(sp['X_tr'],sp['y_cls_tr'],eval_set=[(sp['X_va'],sp['y_cls_va'])],
            callbacks=[lgb.early_stopping(EARLY_STOP,verbose=False),lgb.log_evaluation(0)])
    print(f'  Clf best_iter={clf.best_iteration_}')
    return {'reg':reg,'clf':clf}

def evaluate_stability(models, samples, horizon):
    sp=samples[horizon]; reg,clf=models['reg'],models['clf']
    yp=reg.predict(sp['X_te']); yt=sp['y_reg_te']
    rmse=np.sqrt(mean_squared_error(yt,yp)); mae=mean_absolute_error(yt,yp); r2=r2_score(yt,yp)
    ycp=clf.predict(sp['X_te']); yct=sp['y_cls_te']
    acc=accuracy_score(yct,ycp); f1=f1_score(yct,ycp,average='macro')
    print(f'  Reg: RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}')
    print(f'  Clf: Acc={acc:.3f} F1={f1:.3f}')
    print(classification_report(yct,ycp,target_names=['Not-Stable','Stable'],digits=3,zero_division=0))
    return dict(horizon=horizon,RMSE=rmse,MAE=mae,R2=r2,Accuracy=acc,Macro_F1=f1)

print('Helpers loaded')

# ============================================================
# Cell 9: 訓練模型 & 儲存
# ============================================================
stability_trained={}; all_metrics=[]
for h in HORIZONS:
    print(f'\n{"="*50}\n Training t+{h}min\n{"="*50}')
    m=train_stability_models(samples,h); stability_trained[h]=m
    all_metrics.append(evaluate_stability(m,samples,h))
    joblib.dump(m,os.path.join(MODEL_DIR,f'STAB_h{h}.pkl'))
    print(f'  Saved STAB_h{h}.pkl')
report=pd.DataFrame(all_metrics)
report.to_csv(os.path.join(MODEL_DIR,'stability_evaluation_report.csv'),index=False)
print('\n=== Summary ==='); print(report.to_string(index=False))

# ============================================================
# Cell 10: 散佈圖 — 預測 vs 實際
# ============================================================
fig,axes=plt.subplots(1,len(HORIZONS),figsize=(6*len(HORIZONS),5))
if len(HORIZONS)==1: axes=[axes]
for ax,h in zip(axes,HORIZONS):
    sp=samples[h]; yt=sp['y_reg_te']; yp=stability_trained[h]['reg'].predict(sp['X_te'])
    ax.scatter(yt,yp,s=1,alpha=0.15,color='steelblue')
    lims=[-0.3,0.8]; ax.plot(lims,lims,'r--',lw=1)
    for thr,col in [(0.10,'orange'),(0.20,'red')]:
        ax.axhline(thr,color=col,ls=':',lw=0.8); ax.axvline(thr,color=col,ls=':',lw=0.8)
    ax.set_title(f't+{h}min R2={r2_score(yt,yp):.3f}')
    ax.set_xlabel('Actual alpha'); ax.set_ylabel('Predicted alpha')
    ax.set_xlim(lims); ax.set_ylim(lims); ax.grid(alpha=0.3)
plt.suptitle('Stability Index alpha: Predicted vs Actual',fontsize=14,y=1.02)
plt.tight_layout(); plt.savefig(os.path.join(MODEL_DIR,'stability_scatter.png'),dpi=120); plt.show()

# ============================================================
# Cell 11: 時間序列圖 — 穩定度預測
# ============================================================
fig,axes=plt.subplots(len(HORIZONS),1,figsize=(14,3.5*len(HORIZONS)))
if len(HORIZONS)==1: axes=[axes]
for ax,h in zip(axes,HORIZONS):
    sp=samples[h]; yt=sp['y_reg_te']; yp=stability_trained[h]['reg'].predict(sp['X_te'])
    n=min(800,len(yt)); x=range(n)
    ax.plot(x,yt[:n],'b-',lw=0.8,label='Actual',alpha=0.8)
    ax.plot(x,yp[:n],'r-',lw=0.8,label='Predicted',alpha=0.8)
    ax.axhline(STABLE_THRESHOLD,color='red',ls='--',lw=1.2,label=f'Stable threshold={STABLE_THRESHOLD}')
    # removed neutral band (binary now)
    ax.set_ylabel('alpha'); ax.set_title(f't+{h}min Stability Time Series (test, first {n})')
    ax.legend(loc='upper right',fontsize=8); ax.grid(alpha=0.3)
axes[-1].set_xlabel('Sample #')
plt.tight_layout(); plt.savefig(os.path.join(MODEL_DIR,'stability_timeseries.png'),dpi=120); plt.show()

# ============================================================
# Cell 12: 混淆矩陣
# ============================================================
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
fig,axes=plt.subplots(1,len(HORIZONS),figsize=(5*len(HORIZONS),4))
if len(HORIZONS)==1: axes=[axes]
for ax,h in zip(axes,HORIZONS):
    sp=samples[h]; yt=sp['y_cls_te']; yp=stability_trained[h]['clf'].predict(sp['X_te'])
    cm=confusion_matrix(yt,yp)
    ConfusionMatrixDisplay(cm,display_labels=['Not-Stable','Stable']).plot(ax=ax,cmap='Blues',colorbar=False)
    ax.set_title(f't+{h}min Acc={accuracy_score(yt,yp):.3f}')
plt.suptitle('Stability Classification Confusion Matrix',fontsize=14,y=1.02)
plt.tight_layout(); plt.savefig(os.path.join(MODEL_DIR,'stability_confusion.png'),dpi=120); plt.show()

# ============================================================
# Cell 13: 特徵重要性
# ============================================================
h_show=30 if 30 in HORIZONS else HORIZONS[0]
reg=stability_trained[h_show]['reg']
feat_cols=FEATURES+TIME_FEATS
feat_names=[f't-{WINDOW-1-t}_{f}' for t in range(WINDOW) for f in feat_cols]
imp=reg.feature_importances_; top_k=25; idx_top=np.argsort(imp)[-top_k:]
fig,ax=plt.subplots(figsize=(9,7))
ax.barh(range(top_k),imp[idx_top],color='steelblue')
ax.set_yticks(range(top_k)); ax.set_yticklabels([feat_names[i] for i in idx_top],fontsize=8)
ax.set_xlabel('Importance'); ax.set_title(f'Stability Regression Top {top_k} Features (t+{h_show}min)')
for i,idx in enumerate(idx_top):
    name=feat_names[idx]
    for sf in STABILITY_FEATS:
        if sf in name:
            ax.get_yticklabels()[i].set_color('darkorange')
            ax.get_yticklabels()[i].set_fontweight('bold'); break
ax.annotate('Orange = stability-specific features',xy=(0.55,0.02),xycoords='axes fraction',fontsize=9,color='darkorange')
plt.tight_layout(); plt.savefig(os.path.join(MODEL_DIR,'stability_feature_importance.png'),dpi=120); plt.show()

# ============================================================
# Cell 14: 推論示範
# ============================================================
def predict_stability(df_recent, models, horizon):
    feat_cols=FEATURES+TIME_FEATS
    x=df_recent[feat_cols].values.astype(np.float32)[-WINDOW:]
    if len(x)<WINDOW: return None
    X=x.ravel().reshape(1,-1)
    ap=models['reg'].predict(X)[0]; cp=models['clf'].predict(X)[0]
    cn={0:'Not-Stable',1:'Stable'}
    return dict(alpha=ap,cls=cp,cls_name=cn[cp],horizon=horizon)

print('=== Inference Demo ===')
last=df.iloc[-WINDOW-10:]
for h in HORIZONS:
    r=predict_stability(last,stability_trained[h],h)
    if r: print(f'  t+{h}min: alpha={r["alpha"]:.3f} -> {r["cls_name"]}')

# ============================================================
# Parameter Guide
# ============================================================
# ## Alpha Thresholds (Cell 3)
# ALPHA_THRESHOLDS = {'unstable': 0.10, 'stable': 0.20}
# Adjust based on the distribution plot in Cell 6.
#
# ## Power Law Formula
# WS(z) = WS_ref * (z/z_ref)^alpha
# ln(WS) = alpha * ln(z) + c
# -> OLS on 3 heights (38m, 69m, 100m) gives alpha
