# 📊 Quant Portfolio Analyzer 技術架構與開發文件

本文件詳細說明 `quant-portfolio-analyzer` 的系統架構、技術棧與核心邏輯，供日後開發與維護參考。本文件內容已經過財務金融學原理之驗證，確保回測與量化指標計算符合學術與業界標準。

---

## 🛠️ 技術棧 (Tech Stack)

### 1. 核心開發環境
- **語言**: Python 3.9+
- **網頁框架**: [Streamlit](https://streamlit.io/) (用於快速建構互動式資料應用程式)

### 2. 資料獲取與處理
- **yfinance**: 串接 Yahoo Finance API，獲取全球市場歷史價格與 ETF 基本面資料。包含 `auto_adjust=True` 取回還原股價。
- **Pandas & NumPy**: 核心資料結構與矩陣運算，處理時間序列 (Time-series) 資料，與跨國時間軸對齊。
- **Requests**: 用於抓取 Wikipedia (S&P 500 名單)、證交所 OpenAPI 與元大投信 API (台股成分股)。

### 3. 量化運算與統計
- **SciPy (stats)**: 用於執行線性迴歸 (OLS)，計算事件分析中的 Beta 值與異常報酬。
- **數理財務指標計算**: 實作 CAGR、夏普比率 (Sharpe)、卡瑪比率 (Calmar)、下檔捕獲率 (Downside Capture) 與動態滾動報酬演算法。

### 4. 資料視覺化
- **Plotly (Graph Objects)**: 產生高互動性的金融圖表，包含多線走勢圖、水下圖 (Underwater Chart)、雷達圖、甜甜圈圖。

---

## 🏗️ 系統架構 (Architecture)

專案採 **模組化設計**，將 UI 渲染與核心計算分離：

### 1. 入口文件：`app.py`
- **職責**: 導航中心 (Tab 系統)、全域狀態管理與各功能整合。
- **分頁結構**:
    - **📊 量化分析**: 主力回測引擎，支援自訂權重與動態滾動視窗。
    - **🦅 美股掃描** / **🇹🇼 台股掃描**: 調用 `market_screener.py` 展示市場矩陣。
    - **⚔️ ETF 擂台**: 單一資產之基本面與績效對決。
    - **🌱 定期定額**: DCA 複利計算模擬器。

### 2. 事件分析模組：`event_study.py`
- **職責**: 分析重大事件前後的超額表現。
- **邏輯 (Market Model)**: 
    1. 抓取事件視窗 (Event Window) 數據。
    2. 使用 **市場模型 (Market Model)** 迴歸出大盤的 Alpha 與 Beta。
    3. 扣除大盤預期報酬，求得純淨的 **異常報酬 (Abnormal Return, AR)** 與 **累積異常報酬 (Cumulative AR, CAR)**。

### 3. 市場掃描模組：`market_screener.py`
- **職責**: 動態資料爬蟲與市場矩陣渲染。
- **核心功能**: 實作自動爬取維基百科與公開資料，並透過 `calculate_multi_timeframe_returns` 計算多重時間窗報酬。

---

## 🔬 核心技術與財務邏輯深鑽 (Data Fetching & Quantitative Logic)

此部分詳述專案內與 API 串接、資料預處理以及財務指標運算相關之公式與邏輯，確保其無偏誤且符合財金標準。

### 1. 資料抓取架構與防偏誤設計 (Data APIs)
- **多來源名單動態爬取 (避免倖存者偏差)**：
    - **S&P 500 (美股)**：透過 Python `requests` 與 `pandas.read_html` 動態解析 **Wikipedia** 的 S&P 500 最新成分股表格，確保不在名單上的過氣公司會被正確汰除。
    - **台灣 50 (台股)**：
        - 呼叫 **元大投信 API** (`yuantaetfs.com`) 即時取得 0050 ETF 的最新成分股與權重。
        - 呼叫 **台灣證券交易所 OpenAPI** 獲取官方且標準化的產業分類，用以分門別類。
- **資本利得與現金股利還原機制**：
    - 呼叫 `yfinance` 獲取歷史價格時，強制設定 `auto_adjust=True`。這使得回傳的收盤價是經過除權息還原的 **Adjusted Close**。
    - 財務正確性意義在於：計算報酬率時若不還原股價，會因為發放現金股利造成圖表上的「虛假崩跌」。還原後所計算的是財金界標準的 **「總報酬 (Total Return)」**。

### 2. Market Screener 運算邏輯
- **多時間窗持有期間報酬率 (Holding Period Return, HPR)**：
    - **公式**：$\text{HPR} = (\frac{P_{today}}{P_{t-n}} - 1) \times 100\%$
    - **交易日轉換規則**：金融市場排除假日，因此程式中天數參數符合學界約定：
        - 1 個月 (1mo) $\approx 21$ 個交易日
        - 3 個月 (3mo) $\approx 63$ 個交易日
        - 6 個月 (6mo) $\approx 126$ 個交易日
        - 1 年 (1y) $\approx 252$ 個交易日
    - **YTD (Year-to-Date)**：鎖定並提取當年度第一個交易日的還原價格作為分母。

### 3. 量化分析核心指標公式 (Portfolio Metrics in app.py)
在 `app.py` 的 `calculate_metrics` 函式中，實作了以下財務指標：
- **累積報酬 (Cumulative Return)**：
  - 將每日的百分比報酬轉化為複利乘積序列 `(1 + r_t).cumprod()`。
- **年化報酬率 (CAGR - Compound Annual Growth Rate)**：
  - $\text{CAGR} = (1 + \text{Total Return})^{\frac{1}{N}} - 1$
  - 其中 $N$ 為實際交易日總數除以 252。使用幾何平均而非算術平均，正確反映了資產長期複利的縮水/增長效應。
- **年化波動率 (Annualized Volatility)**：
  - 由每日報酬的樣本標準差乘上 $\sqrt{252}$ 而來，用以代表總風險。
- **夏普比率 (Sharpe Ratio)**：
  - $\text{Sharpe} = \frac{\text{CAGR} - R_f}{\text{Annualized Volatility}}$
  - 系統內建無風險利率 $R_f = 3\%$，代表每承擔一單位的年化風險，能換取多少的超額報酬。
- **最大回撤 (Maximum Drawdown, MDD)**：
  - 演算法追蹤該序列在過去任意時間點的波峰 (`running_max = cumulative.cummax()`)。
  - 計算 $\frac{\text{Current Value} - \text{Running Max}}{\text{Running Max}}$ 的極小值，測出最糟時期的帳面跌幅。
- **卡瑪比率 (Calmar Ratio)**：
  - $\text{Calmar} = \frac{\text{CAGR}}{|\text{MDD}|}$ 
  - 避險基金常用的「抗跌效能」指標，衡量每承受 1% 的回撤風險所帶來的年化報酬。
- **下檔捕獲率 (Downside Capture Ratio)**：
  - 防禦屬性分析：由 Pandas 遮罩 (`benchmark_returns < 0`) 過濾出大盤下跌的交易日。
  - 將那幾天大盤的跌幅組合起來，與你的投資組合在同幾天的跌幅相比。小於 1 (100%) 代表你的資產具有防禦力。

### 4. 跨國資料對齊與前向填補機制 (Cross-Border Sync & Forward Fill)
處理台、美股交易日不同的核心挑戰：
```python
# app.py 與 market_screener.py 內的防呆對齊
df_compare = df_compare.ffill().dropna(how='all')
returns = df_compare.pct_change().fillna(0)
```
- 因時區與國定假日不同 (如台灣春節休市，美股照開)，若強行合併（Join）會產生 NaN 空值。
- `ffill()` (Forward Fill)：將休市日的缺漏值以「前一個交易日之收盤價」延續填補。
- 這能確保：
    1. 矩陣維度完全對齊，迴歸分析與加權相加不會因 `NaN` 而報錯。
    2. 對於休市的市場，單日報酬率藉此會被算為 **0%**，使整體淨值曲線維持水平，不會扭曲累積結果。

---

## 🚀 未來擴充方向 (Future Improvements)
1. **多幣別自動轉換**: 目前假設各資產幣別一致，未來可導入即時匯率 API 進行換算。
2. **更多風險分析**: 引入 Sortino Ratio (考量下方波動) 或 VaR (Value at Risk) 分析。
3. **資料庫串接**: 將爬取的成分股名單本地化儲存與設計 Redis 快取，進一步提升載入速度。
