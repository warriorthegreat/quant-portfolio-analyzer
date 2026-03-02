import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import plotly.graph_objects as go

# ==========================================
#  共用設定與常數 (字典與對照表)
# ==========================================
PERIOD_OPTIONS = {
    "近 1 個月 (短期動能)": "1mo",
    "近 3 個月 (季報發酵)": "3mo",
    "近 6 個月 (半年趨勢)": "6mo",
    "今年以來 YTD (年度總結)": "ytd",
    "近 1 年 (長期趨勢)": "1y"
}

# 🦅 美股 GICS 產業中文翻譯對照表
GICS_SECTOR_MAP = {
    "Information Technology": "💻 資訊科技",
    "Health Care": "💊 醫療保健",
    "Financials": "🏦 金融",
    "Consumer Discretionary": "🛍️ 非必需消費",
    "Communication Services": "📡 通訊服務",
    "Industrials": "⚙️ 工業",
    "Consumer Staples": "🛒 必需消費",
    "Energy": "⚡ 能源",
    "Utilities": "💡 公用事業",
    "Real Estate": "🏢 房地產",
    "Materials": "🏭 原物料"
}

# 🇹🇼 台股 TWSE 官方產業代碼/名稱對照表 (防呆機制)
TWSE_SECTOR_MAP = {
    "01": "水泥工業", "02": "食品工業", "03": "塑膠工業", "04": "紡織纖維",
    "05": "電機機械", "06": "電器電纜", "07": "化學工業", "08": "玻璃陶瓷",
    "09": "造紙工業", "10": "鋼鐵工業", "11": "橡膠工業", "12": "汽車工業",
    "13": "電子工業", "14": "建材營造", "15": "航運業", "16": "觀光餐旅",
    "17": "🏦 金融保險業", "18": "貿易百貨", "20": "其他業", "21": "化學工業",
    "22": "💊 生技醫療業", "23": "油電燃氣業", "24": "💻 半導體業", "25": "電腦及週邊設備業",
    "26": "光電業", "27": "通信網路業", "28": "電子零組件業", "29": "電子通路業",
    "30": "資訊服務業", "31": "其他電子業", "32": "文化創意業", "33": "農業科技業",
    "34": "電子商務業", "35": "♻️ 綠能環保業", "36": "數位雲端業", "37": "運動休閒業",
    "38": "居家生活業"
}

# ==========================================
#  核心共用演算法：多時間窗報酬率計算器
# ==========================================
def calculate_multi_timeframe_returns(prices):
    """將股價陣列切片，同時算出 1M, 3M, 6M, YTD, 1Y 的報酬率"""
    if len(prices) == 0:
        return {k: 0 for k in ['1mo', '3mo', '6mo', 'ytd', '1y']}
        
    current_price = prices.iloc[-1]
    
    # 防呆設計：如果股票上市時間不夠長，就拿第一天的價格當基準
    def get_hist_price(days_back):
        return prices.iloc[-(days_back+1)] if len(prices) > days_back else prices.iloc[0]
        
    res = {}
    res['1mo'] = (current_price / get_hist_price(21) - 1) * 100   # 約 21 個交易日
    res['3mo'] = (current_price / get_hist_price(63) - 1) * 100   # 約 63 個交易日
    res['6mo'] = (current_price / get_hist_price(126) - 1) * 100  # 約 126 個交易日
    res['1y']  = (current_price / prices.iloc[0] - 1) * 100
    
    # 計算 YTD (今年以來)
    current_year = prices.index[-1].year
    ytd_prices = prices[prices.index.year == current_year]
    res['ytd'] = (current_price / ytd_prices.iloc[0] - 1) * 100 if len(ytd_prices) > 0 else 0
        
    return res

def get_trend_data(prices, period_code):
    """根據使用者下拉選單的週期，裁切走勢圖的長度"""
    days_map = {'1mo': 21, '3mo': 63, '6mo': 126, '1y': 252}
    if period_code in days_map:
        days = days_map[period_code]
        return prices.iloc[-days:].tolist() if len(prices) > days else prices.tolist()
    elif period_code == 'ytd':
        current_year = prices.index[-1].year
        return prices[prices.index.year == current_year].tolist()
    return prices.tolist()


# ==========================================
#  模組 1：🦅 美股 S&P 500 核心邏輯
# ==========================================
@st.cache_data(ttl=3600)
def load_sp500_dashboard(period="1mo"):
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    df_sp500 = pd.read_html(response.text)[0]
    df_sp500['Symbol'] = df_sp500['Symbol'].str.replace('.', '-')
    tickers = df_sp500['Symbol'].tolist()
    
    # 🎯 關鍵改動：不管使用者選什麼，底層一律先抓 1 年資料，用來算多個時間窗
    prices_raw = yf.download(tickers, period="1y", auto_adjust=True, progress=False, threads=True)
    
    if isinstance(prices_raw.columns, pd.MultiIndex) and 'Close' in prices_raw.columns.levels[0]:
        close_prices = prices_raw['Close']
    elif 'Close' in prices_raw.columns:
        close_prices = prices_raw['Close']
    else:
        close_prices = prices_raw 
        
    close_prices = close_prices.ffill().dropna(how='all')
    actual_start = close_prices.index.min().strftime('%Y-%m-%d')
    actual_end = close_prices.index.max().strftime('%Y-%m-%d')
        
    results = []
    for idx, row in df_sp500.iterrows():
        ticker = row['Symbol']
        if ticker in close_prices.columns:
            stock_prices = close_prices[ticker].dropna()
            if len(stock_prices) > 0:
                returns = calculate_multi_timeframe_returns(stock_prices)
                trend = get_trend_data(stock_prices, period)
                
                # 美股產業中文翻譯
                eng_sector = row['GICS Sector']
                cht_sector = GICS_SECTOR_MAP.get(eng_sector, eng_sector)
                
                results.append({
                    '代號': ticker,
                    '公司名稱': row['Security'],
                    '產業板塊': cht_sector,
                    '區間報酬率': returns[period], # 供長條圖統計用
                    '1M 報酬': returns['1mo'],
                    '3M 報酬': returns['3mo'],
                    '6M 報酬': returns['6mo'],
                    '區間走勢': trend 
                })
    return pd.DataFrame(results), actual_start, actual_end

# ==========================================
#  模組 2：🇹🇼 台股 50 核心邏輯
# ==========================================
@st.cache_data(ttl=86400)
def get_twse_official_info():
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        twse_dict = {}
        for company in data:
            ticker = company.get("公司代號")
            name = company.get("公司簡稱", "未知名稱")
            sector_raw = company.get("產業別", "其他")
            
            # 台股產業中文翻譯 (若 API 給代碼如 "24"，會轉成 "半導體業"；若給字串則不變)
            sector = TWSE_SECTOR_MAP.get(sector_raw, sector_raw)
            
            if ticker:
                twse_dict[ticker] = {"name": name, "sector": sector}
        return twse_dict
    except Exception as e:
        return {}

@st.cache_data(ttl=86400)
def get_dynamic_tw50_tickers():
    url = "https://www.yuantaetfs.com/api/StkWeights?date=&fundid=1066"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        data = res.json()
        return [item['code'] + ".TW" for item in data]
    except Exception:
        return ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2881.TW", "2882.TW", "2891.TW", "2886.TW", "2412.TW", "1216.TW", "3231.TW", "3711.TW", "2884.TW", "2002.TW", "5871.TW", "3034.TW", "2892.TW", "2303.TW", "2885.TW", "2890.TW", "2883.TW", "2887.TW", "2345.TW", "2324.TW", "2379.TW", "2301.TW", "2357.TW", "3045.TW", "2395.TW", "2912.TW", "2880.TW", "1101.TW", "1301.TW", "1303.TW", "2603.TW", "2609.TW", "2615.TW", "1590.TW", "2207.TW", "5880.TW", "6669.TW", "2356.TW", "4938.TW", "2385.TW", "3008.TW", "2408.TW", "2353.TW", "1326.TW", "1102.TW"]

@st.cache_data(ttl=3600)
def load_tw_dashboard(period="1mo"):
    tw50_tickers = get_dynamic_tw50_tickers()
    TWSE_INFO = get_twse_official_info()
    
    # 一樣無條件先抓 1 年資料供多重時間窗計算
    prices_raw = yf.download(tw50_tickers, period="1y", auto_adjust=True, progress=False, threads=True)
    
    if isinstance(prices_raw.columns, pd.MultiIndex) and 'Close' in prices_raw.columns.levels[0]:
        close_prices = prices_raw['Close']
    elif 'Close' in prices_raw.columns:
        close_prices = prices_raw['Close']
    else:
        close_prices = prices_raw 
        
    close_prices = close_prices.ffill().dropna(how='all')
    actual_start = close_prices.index.min().strftime('%Y-%m-%d')
    actual_end = close_prices.index.max().strftime('%Y-%m-%d')
        
    results = []
    for ticker in tw50_tickers:
        if ticker in close_prices.columns:
            stock_prices = close_prices[ticker].dropna()
            if len(stock_prices) > 0:
                returns = calculate_multi_timeframe_returns(stock_prices)
                trend = get_trend_data(stock_prices, period)
                
                pure_ticker = ticker.replace('.TW', '')
                company_info = TWSE_INFO.get(pure_ticker, {"name": f"代號 {pure_ticker}", "sector": "其他業"})
                
                results.append({
                    '代號': pure_ticker,
                    '公司名稱': company_info["name"],
                    '產業板塊': company_info["sector"],
                    '區間報酬率': returns[period], # 供長條圖統計用
                    '1M 報酬': returns['1mo'],
                    '3M 報酬': returns['3mo'],
                    '6M 報酬': returns['6mo'],
                    '區間走勢': trend 
                })
    return pd.DataFrame(results), actual_start, actual_end

# ==========================================
#  模組 3：共用 UI 渲染引擎
# ==========================================
def render_market_dashboard(title, description, market_type):
    st.title(title)
    st.markdown(description)
    
    prefix = market_type
    
    p_col1, p_col2 = st.columns([1, 2])
    with p_col1:
        selected_period_label = st.selectbox(
            f"📊 選擇「板塊強弱長條圖」及「走勢圖」的統計週期：", 
            list(PERIOD_OPTIONS.keys()), 
            key=f"{prefix}_period"
        )
        selected_period_code = PERIOD_OPTIONS[selected_period_label]

    state_key = f"{prefix}_loaded"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    if st.button(f"🔄 載入最新掃描數據", key=f"{prefix}_btn"):
        st.session_state[state_key] = True

    if st.session_state[state_key]:
        with st.spinner(f"🚀 正在批次計算多重時間窗與市場資料，請稍候..."):
            
            if market_type == "sp500":
                df_dash, start_dt, end_dt = load_sp500_dashboard(period=selected_period_code)
            else:
                df_dash, start_dt, end_dt = load_tw_dashboard(period=selected_period_code)
            
            st.success("✅ 載入成功！")
            
            st.subheader("🎯 深入挖掘：板塊成分股篩選")
            all_sectors = sorted(df_dash['產業板塊'].unique().tolist())
            
            selected_sectors = st.multiselect(
                "請選擇你想查看的產業板塊（可複選）：", 
                options=all_sectors, 
                default=all_sectors,
                key=f"{prefix}_multiselect"
            )
            filtered_df = df_dash[df_dash['產業板塊'].isin(selected_sectors)]

            if not filtered_df.empty:
                f_total = len(filtered_df)
                f_up = (filtered_df['區間報酬率'] > 0).sum()
                f_down = f_total - f_up
                
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("篩選成分股總數", f_total)
                fc2.metric(f"上漲家數 ({selected_period_label})", f"📈 {f_up} 家")
                fc3.metric(f"下跌家數 ({selected_period_label})", f"📉 {f_down} 家")
                
                st.markdown("<br>", unsafe_allow_html=True) 

                # 💡 加上 include_groups=False 完美消滅 Pandas 棄用警告！
                sector_summary = filtered_df.groupby('產業板塊').apply(
                    lambda x: pd.Series({
                        '上漲家數': (x['區間報酬率'] > 0).sum(),
                        '下跌家數': (x['區間報酬率'] <= 0).sum()
                    }),
                    include_groups=False 
                ).reset_index()
                
                sector_summary['總家數'] = sector_summary['上漲家數'] + sector_summary['下跌家數']
                sector_summary = sector_summary.sort_values('總家數', ascending=True)

                fig_sector = go.Figure()
                fig_sector.add_trace(go.Bar(
                    y=sector_summary['產業板塊'], x=sector_summary['下跌家數'],
                    name='下跌家數', orientation='h', marker=dict(color='#ff4b4b') 
                ))
                fig_sector.add_trace(go.Bar(
                    y=sector_summary['產業板塊'], x=sector_summary['上漲家數'],
                    name='上漲家數', orientation='h', marker=dict(color='#00cc96') 
                ))
                
                dynamic_height = max(250, len(sector_summary) * 45) 
                fig_sector.update_layout(
                    title=f"各產業板塊漲跌家數統計 (依據：{selected_period_label})",
                    barmode='stack', height=dynamic_height, bargap=0.4, 
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_sector, use_container_width=True)
            else:
                st.warning("⚠️ 請至少選擇一個產業板塊來顯示圖表。")

            st.divider()
            
            st.subheader("⚡ 動能共振掃描矩陣 (Momentum Resonance Matrix)")
            st.markdown("觀察 1M / 3M / 6M 的報酬率變化，尋找長多短空 (洗盤) 或長中短皆強 (多頭共振) 的標的。")

            # 🎯 關鍵改動：調整資料表的欄位順序與顯示方式
            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=600,
                column_order=["代號", "公司名稱", "產業板塊", "1M 報酬", "3M 報酬", "6M 報酬", "區間走勢"],
                column_config={
                    "代號": st.column_config.TextColumn("代號", width="small"),
                    "公司名稱": st.column_config.TextColumn("公司名稱", width="small"),
                    "產業板塊": st.column_config.TextColumn("產業板塊", width="medium"),
                    "1M 報酬": st.column_config.NumberColumn("1M 報酬 (4W)", format="%.2f%%"),
                    "3M 報酬": st.column_config.NumberColumn("3M 報酬 (13W)", format="%.2f%%"),
                    "6M 報酬": st.column_config.NumberColumn("6M 報酬 (26W)", format="%.2f%%"),
                    "區間走勢": st.column_config.LineChartColumn(f"走勢圖 ({selected_period_label})", y_min=0, width="medium")
                }
            )