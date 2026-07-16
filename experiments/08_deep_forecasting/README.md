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
2.  `results/`：儲存評估結果與對比圖表的目錄。
