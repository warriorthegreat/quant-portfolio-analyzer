# 📊 Quant Portfolio Analyzer (雙博士投資組合分析儀)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://你的網址.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本專案是一個基於 Python 與 Streamlit 開發的專業級量化投資回測與風險評估平台。旨在幫助投資人跳脫單一時間點的後見之明，透過動態滾動視窗與高階風險指標，真實評估資產配置在不同市場週期下的勝率與抗震能力。

---

## 👨‍💻 關於開發者 (About the Author)

Alvin Zhang | 國立台灣大學 圖書資訊學系研究所 (NTU LIS) 
🔗 [LinkedIn Profile](https://www.linkedin.com/in/kun-jie-zhang-376902284/)

> "Quantitative finance is essentially applied Information Science."
> 
> 作為圖資系研究生，我將資訊檢索 (Information Retrieval) 與資料清理 (Data Cleaning) 的學術訓練，應用於複雜的跨國金融時間序列數據中。本系統不僅是一個財務計算機，更是將非結構化市場數據轉化為「可決策資訊 (Actionable Insights)」的具體實踐。

---

## 🌟 核心功能 (Core Features)

### 1. 動態滾動報酬與勝率分析 (Dynamic Rolling Returns)
告別特定起點起算的運氣偏差。系統引入華爾街標準的滑動視窗演算法，支援 1年期、3年期、5年期 動態切換：
* 歷史勝率 (Win Rate)：計算在過去任何交易日進場並持有 N 年的絕對正報酬機率。
* 超額勝率 (Beat Market Rate)：評估策略打敗大盤的真實機率。

### 2. 高階風險指標 (Advanced Risk Metrics)
不只看報酬，更重視風險調整後的表現：
* 夏普比率 (Sharpe Ratio)：衡量承擔每單位總風險所產生的超額報酬。
* 卡瑪比率 (Calmar Ratio)：評估策略的抗跌效能，數值愈高代表從回撤中恢復的能力愈強。
* 下檔捕獲率 (Downside Capture)：防禦裝甲測試，計算大盤下跌時，投資組合的連動跌幅比例。

### 3. 跨國日曆自動對齊 (Cross-Border Holiday Misalignment Fix)
解決台股 (如 0050.TW) 與美股 (如 VOO, SPY) 因國定假日不同導致的資料錯位問題。系統底層實作 ffill() 前向填補機制，確保跨國資產比較時的時間序列陣列完全 1:1 對齊。

### 4. ETF 擂台與基本面爬蟲 (ETF Arena & Web Scraping)
支援兩檔標的之一對一單挑，透過 yfinance 實時串接 Yahoo Finance API，抓取 AUM (資產規模)、殖利率與 52 週高低點。

---

## 🛠️ 技術棧 (Tech Stack)

* 前端框架: Streamlit (快速建立互動式資料型應用程式)
* 資料處理: Pandas, NumPy (時間序列處理、矩陣運算)
* 視覺化: Plotly (高互動性金融圖表、水下圖、雷達圖)
* 數據源: yfinance (Yahoo Finance API 串接)

---

## 🚀 如何在本地端執行 (Local Installation)

1. Clone 本專案：
   ```bash
   git clone [https://github.com/你的帳號/你的儲存庫.git](https://github.com/你的帳號/你的儲存庫.git)
   cd 你的儲存庫
   
2. 安裝所需套件：
```bash
pip install -r requirements.txt

3. 啟動 Streamlit 伺服器：
```bash
streamlit run app.py
