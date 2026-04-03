# 📊 Quant Portfolio Analyzer 技術架構與開發文件

本文件詳細說明 `quant-portfolio-analyzer` 的系統架構、技術棧與核心邏輯，供日後開發與維護參考。

---

## 🛠️ 技術棧 (Tech Stack)

### 1. 核心開發環境
- **語言**: Python 3.9+
- **網頁框架**: [Streamlit](https://streamlit.io/) (用於快速建構互動式資料應用程式)

### 2. 資料獲取與處理
- **yfinance**: 串接 Yahoo Finance API，獲取全球市場歷史價格與 ETF 基本面資料。
- **Pandas & NumPy**: 核心資料結構與矩陣運算，處理時間序列 (Time-series) 資料。
- **Requests**: 用於抓取 Wikipedia (S&P 500 名單)、證交所 OpenAPI 與元大投信 API (台股成分股)。

### 3. 量化運算與統計
- **SciPy (stats)**: 用於執行線性迴歸 (OLS)，計算事件分析中的 Beta 值與異常報酬。
- **自行開發指標**: 實作 CAGR、夏普比率 (Sharpe)、卡瑪比率 (Calmar)、下檔捕獲率 (Downside Capture) 與動態滾動報酬演算法。

### 4. 資料視覺化
- **Plotly (Graph Objects)**: 產生高互動性的金融圖表，包含：
    - 多線走勢圖 (Wealth Index)
    - 區間充填圖 (Underwater Chart)
    - 雷達圖 (Radar Chart)
    - 甜甜圈圖 (Pie Chart)

---

## 🏗️ 系統架構 (Architecture)

專案採 **模組化設計**，將 UI 渲染與核心計算分離：

### 1. 入口文件：`app.py`
- **職責**: 導航中心 (Tab 系統)、全域狀態管理與各功能整合。
- **分頁結構**:
    - **📊 量化分析**: 主力回測引擎，支援自訂權重。
    - **🦅 美股掃描**: 調用 `market_screener.py` 展示 S&P 500 數據。
    - **🇹🇼 台股掃描**: 展示台灣 50 成分股數據。
    - **⚔️ ETF 擂台**: 兩檔標的之一對一基本面與績效對決。
    - **🌱 定期定額**: DCA 複利計算模擬器。

### 2. 事件分析模組：`event_study.py`
- **Class**: `EventStudyAnalyzer`
- **邏輯**: 
    1. 抓取事件前後數據。
    2. 使用 **市場模型 (Market Model)** 計算預期報酬。
    3. 扣除大盤影響後，算出 **異常報酬 (AR)** 與 **累積異常報酬 (CAR)**。

### 3. 市場掃描模組：`market_screener.py`
- **職責**: 動態資料爬蟲與市場矩陣渲染。
- **核心功能**: 
    - 自動從開放資料獲取最新成分股名單。
    - `calculate_multi_timeframe_returns`: 一次性計算 1M, 3M, 6M, YTD 等多重時間窗報酬。
    - `render_market_dashboard`: 共用 UI 引擎，負責渲染長條圖與資料表。

---

## 🔬 核心技術亮點 (Key Implementation Logic)

### 1. 跨國資料對齊機制 (Cross-Border Sync)
針對台、美股交易日不同的問題，系統在底層採用了以下機制：
```python
# 透過 ffill() 前向填補缺值（例如美股開盤而台股休市），確保對位。
df_compare = df_compare.ffill().dropna(how='all')
```

### 2. 效能優化：快取 (Caching)
大量使用 `st.cache_data` 減少重複的 API 請求與運算：
- 股價資料: TTL = 1 小時。
- 市場名單: TTL = 24 小時。

### 3. 動態滾動演算法
不同於傳統的起點分析，系統實作滑動視窗 (Sliding Window)：
- 計算在歷史中任何一天進場持有 N 年的「真實勝率」，消除特定時間點的選擇偏差。

---

## 🚀 未來擴充方向 (Future Improvements)
1. **多幣別自動轉換**: 目前假設各資產幣別一致，未來可導入匯率換算。
2. **更多風險分析**: 引入 Sortino Ratio 或 VaR (Value at Risk) 分析。
3. **資料庫串接**: 將爬取的成分股名單本地化儲存，進一步提升載入速度。
