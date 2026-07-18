# 實驗 08 | BSMI 風場深度學習時序預報 (DLinear Model)
BSMI Wind Speed Long-Term Time-Series Forecasting via DLinear Model

本實驗室專注於使用 SOTA 的輕量化深度時序網路——**DLinear**，解決高頻風速的多步長期預報任務。

---

## 1. 為什麼選擇 DLinear？
在先前 **實驗 07 (VMD 訊號分解)** 中，我們發現雖然變分模態分解在全量分解上表現驚人，但在**嚴格無洩漏的即時滾動預報**中，會因為**端點效應 (Edge Effects)** 產生劇烈的邊界噪訊，使得 VMD-LGBM/RF 的預估精度反而不如直接預估模型。

為了解決這個瓶頸，DLinear 被提出：
1. **內置分解機制 (Embedded Decomposition)**：不再將時頻分解作為預處理（Preprocessing），而是直接將「趨勢-季節分解」做為神經網路的一層（Layer），在反向傳播中自適應優化。
2. **無端點變形**：DLinear 的線性映射（Linear Layer）直接對時間步進行加權，對邊界抖動不敏感，顯著抑制了端點雜訊。
3. **無未來洩漏 (Zero Leakage)**：在 PyTorch 時序滑動視窗中，歷史 $t$ 的特徵只映射到未來的 $t+1 \sim t+H$，完全符合因果關係。

---

## 2. 實驗配置與參數
*   **輸入特徵**：過去 $Seq\_Len = 144$ 點 (24 小時的 100m 風速歷史)。
*   **預測標籤**：未來 $Pred\_Len = 12$ 點 (2 小時的 100m 風速軌跡，直接輸出，非自迴歸迭代)。
*   **數據集分割**：前 70% 用於模型訓練，後 30% 用於模型測試與效能評估。
*   **分解窗口 (Decomp Kernel)**：$25$。

---

## 3. 執行與評估
您可以直接執行主程式進行 PyTorch 模型的訓練與評估：
```bash
python wind_dlinear_forecasting.py
```

### 輸出結果與分析
執行完成後，指標與對比圖將自動輸出至以下路徑：
1.  `results/dlinear_forecast.png`：實際風速與 DLinear 在 t+10m 預報軌跡的對比圖。
2.  `results/dlinear_metrics.csv`：DLinear 在未來各個時間步 (t+10m 至 t+120m) 的 MAE, RMSE, $R^2$ 評估指標。

---

## 4. 檔案說明
1.  `wind_dlinear_forecasting.py`：包含模型定義、資料處理、訓練循環與指標畫圖的主程式。
2.  `wind_comparison_forecasting.py`：包含多模型對比訓練 (DLinear, NLinear, PatchLinear, LightGBM) 的比較程式。
3.  `results/`：儲存評估結果與對比圖表的目錄。

---

## 5. SOTA 模型與新一代機器學習對比實驗

為了進一步尋求風速預估的最佳架構，我們實現了多模型對比腳本 `wind_comparison_forecasting.py`，比較了以下新一代時序預報模型：

1.  **DLinear**：時序雙線性分解模型。
2.  **NLinear (AAAI 2023)**：去中心化線性模型，特別適合處理風速非平穩（Non-stationary）特徵，防範分佈偏移。
3.  **PatchLinear (2023)**：引入 PatchTST 的 Patching 機制，將 144 點時序切分為重疊片段以捕捉局部語義並過濾隨機噪訊。
4.  **LightGBM (Multi-Output)**：現代梯度提升決策樹代表，以 144 步滯後特徵 (Lagged Features) 進行直接多步預測。

### 測試集指標對比 (t+10m 與 t+120m)

| 模型名稱 (Model) | 預估步長 (Horizon) | MAE (m/s) | RMSE (m/s) | $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| **DLinear** | t+10min | **0.4449** | **0.5679** | **0.9414** |
| **DLinear** | t+120min | **1.6138** | **2.0862** | **0.2635** |
| **NLinear** | t+10min | 0.4499 | 0.5709 | 0.9408 |
| **NLinear** | t+120min | 1.6253 | 2.0974 | 0.2555 |
| **PatchLinear** | t+10min | 0.5752 | 0.7518 | 0.8973 |
| **PatchLinear** | t+120min | 1.7138 | 2.1862 | 0.1912 |
| **LightGBM** | t+10min | 0.5011 | 0.6357 | 0.9266 |
| **LightGBM** | t+120min | 2.3066 | 2.7890 | -0.3164 |

### 結果分析與洞察

1.  **線性映射的高效性**：DLinear 與 NLinear 展現了最強的超短期預測效能 ($R^2$ 達 0.94)。這證明了對於風速預測而言，直接的時間步線性權重映射能有效避免複雜非線性模型的過擬合問題。
2.  **非平穩性的消除**：DLinear 的分解結構與 NLinear 的去中心化手段在此數據段上表現非常接近且皆極佳，表明了在短期視窗內去偏移能提供高度穩健的預測能力。
3.  **Patch 機制的潛力與限制**：PatchLinear ($R^2$ 0.8973) 表現雖略遜於 DLinear/NLinear，但仍取得了很高的準確性。在歷史輸入長度較短的情況下，Patching 減少了線性層的參數數量，對高頻隨機風速噪訊有一定的平滑去噪效果。
4.  **機器學習 (LightGBM) 的極限**：LightGBM 在超短期 $t+10\text{min}$ 預估表現良好 ($R^2 = 0.9266$)，但其在長步長 $t+120\text{min}$ 時出現顯著退化 ($R^2 = -0.3164$)，說明多步直接輸出的 Tabular 模型容易在長步長預測中遺失時間趨勢，對比之下，深度線性時序模型能更好地延續物理時間趨勢。

---

## 6. 新增對比產出
1.  `results/deep_models_comparison.png`：四種模型在測試集前 150 點預測軌跡對比。
2.  `results/deep_models_comparison_metrics.csv`：各模型在不同 Horizon 的詳細指標 CSV。

