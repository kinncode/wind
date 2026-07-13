# BSMI 風場機器學習分析專案 (BSMI Wind Farm Machine Learning Project)

本專案利用位於台灣海峽風場之 **BSMI 觀測測風塔五年的 1 分鐘解析度大氣觀測資料**（期間為 2016-03 至 2021-10），進行一系列機器學習與數據分析實驗。

專案核心涵蓋：風速迴歸預測、風況運行狀態分類、氣候模式分群、大氣穩定度時間序列預報，以及虛擬測風塔（感測器故障備援補值）等五大應用場景。

---

## 📂 專案目錄結構與模組指南

```
c:\kinn\
├── README.md                          ← 本說明文件
├── .gitignore                         ← 排除虛擬環境、模型檔與超大資料檔
├── data/
│   └── BSMI_wind_1min_parquet/        ← 原始 parquet 時序觀測資料（共 64 個檔案，五年期）
│
└── experiments/                       ← 機器學習核心實驗
    ├── 01_wind_regression/            ← 實驗一：風速迴歸預測 (低空預測 100m 輪轂風速)
    │   ├── wind_rf_regression.py
    │   ├── results/                   ← 散布圖、殘差直方圖與基準預測 CSV
    │   └── README.md
    │
    ├── 02_wind_classification/        ← 實驗二：風況四級分類 (低風/正常/強風/極端運行區間)
    │   ├── wind_classification.py
    │   ├── results/                   ← 混淆矩陣、分類報告與分類標籤 CSV
    │   └── README.md
    │
    ├── 03_wind_clustering/            ← 實驗三：風況分群分析 (PCA + K-Means 自動分類典型氣候模式)
    │   ├── wind_clustering.py
    │   ├── results/                   ← 分群雷達圖、極座標風向圖與特徵統計 CSV
    │   └── README.md
    │
    ├── 04_stability_forecast/         ← 實驗四：大氣穩定度預測系統 (未來 2 小時滾動時序預報)
    │   ├── BSMI_Stability_Forecast_CPU_1_local.py
    │   ├── BSMI_Stability_Forecast_*.ipynb
    │   ├── models/                    ← v1, v1_alt, v2 模型與技巧衰減評估圖
    │   └── README.md
    │
    └── 05_virtual_met_mast/           ← 實驗五：虛擬測風塔即時補值 (感測器即時備援系統)
        ├── BSMI_Virtual_Met_Mast.py
        ├── BSMI_Virtual_Met_Mast_Tuning.py
        ├── models/                    ← LightGBM 備援模型與 24小時模擬補值圖
        └── README.md
```

---

## 📊 實驗核心成果總覽

| 實驗模組 | 主要方法 | 預測目標 | 核心效能指標 | 商業價值 / 物理意義 |
|---|---|---|---|---|
| [**01_風速迴歸預測**](file:///c:/kinn/experiments/01_wind_regression/README.md) | 隨機森林迴歸 (RF Regressor) | 100m 平均風速 (`WS_100E`) | **MAE: ~0.1691 m/s**, **$R^2$: 0.992**<br>(比傳統物理 Power Law 物理公式降低 60.5% 誤差) | **降本增效**：利用低高度風速計精準估算高空輪轂風速，降低測風成本。 |
| [**02_風況四級分類**](file:///c:/kinn/experiments/02_wind_classification/README.md) | 隨機森林分類 (RF Classifier) | 四類風況運行狀態 | **Accuracy: 99.62%**, **Macro F1: 99.57%** | **運行預警**：精確判別發電機組處於切入、一般發電、額定滿載或超強風切出。 |
| [**03_風況分群分析**](file:///c:/kinn/experiments/03_wind_clustering/README.md) | PCA + K-Means | 典型氣候模式聚類 ($k=4$) | **自動識別四大氣候型態**：<br>1. 冬季強東北季風 (26.9%)<br>2. 中等強度東北季風 (34.4%)<br>3. 夏季西南季風 (30.3%)<br>4. 夏季微風對流 (8.5%) | **風場規劃**：量化主導風場的典型氣候模式，以進行長期發電效益評估與風機佈署。 |
| [**04_大氣穩定度預報**](file:///c:/kinn/experiments/04_stability_forecast/README.md) | LightGBM (HaF 時序特徵) | 未來 1~120 分鐘大氣穩定度 $\alpha$ | **平均 Accuracy: 94.34%**, **平均 $R^2$: 0.7011**<br>(5-Fold Expanding Window 嚴格驗證) | **安全預警**：提前 2 小時滾動預報大氣穩定度轉換，防止風剪剪應力造成葉片疲勞。 |
| [**05_虛擬測風塔備援**](file:///c:/kinn/experiments/05_virtual_met_mast/README.md) | LightGBM + Optuna 調優 | 100m Anemometer 失效補值 | **MAE: 0.2554 m/s**, **$R^2$: 0.9960**<br>(270萬筆/5年數據五折驗證) | **感測器備援**：當高空主感測器損壞/結冰時，無延遲即時補值，確保 SCADA 持續運行。 |

---

## 🛠️ 環境配置與套件需求

專案預設使用獨立的虛擬環境 `myenv/`，若需自行建立或更新環境，請確保安裝以下套件：

```bash
pip install pandas pyarrow numpy scikit-learn lightgbm xgboost optuna matplotlib seaborn joblib
```

### 支援中文字型繪圖
本專案的 Python 繪圖腳本會自動偵測作業系統中的中文字型（如 Windows 中的微軟正黑體 `Microsoft JhengHei`），若使用無 GUI 的 Linux Server，建議安裝 `Noto Sans CJK TC` 以避免圖表字體顯示為方塊。

---

## 🚀 快速開始指南

每個實驗資料夾內皆包含獨立的 Python 執行腳本，且皆已更新路徑指向根目錄的 `data/`。

### 執行範例

1. **執行虛擬測風塔訓練與補值模擬**：
   ```bash
   cd experiments/05_virtual_met_mast
   python BSMI_Virtual_Met_Mast.py
   ```
   *執行完畢後，產生的模型與視覺化圖表將會自動存放於該目錄下的 `models/` 中。*

2. **執行大氣穩定度時間序列預估**：
   ```bash
   cd experiments/04_stability_forecast
   python BSMI_Stability_Forecast_CPU_1_local.py
   ```

3. **進行超參數調優 (以虛擬測風塔為例)**：
   ```bash
   cd experiments/05_virtual_met_mast
   python BSMI_Virtual_Met_Mast_Tuning.py
   ```
   *此指令會透過 Optuna 搜尋最優參數，尋找結束後會印出最優的參數組合，供主訓練腳本替換。*
