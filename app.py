import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- 網頁設定 ---
st.set_page_config(page_title="雙博士投資組合分析儀 V2.3", layout="wide")

# --- 建立雙分頁 (Tabs) ---
tab1, tab2 = st.tabs(["📊 量化分析 (Analyzer)", "ℹ️ 系統資訊 (About)"])

# ==========================================
#  分頁 2：系統資訊 (About)
# ==========================================
with tab2:
    st.header("ℹ️ 關於本系統")
    st.markdown("""
    **雙博士投資組合分析儀 (Quant Portfolio Analyzer)** * **V2.3 更新：** 加入強大的資料空值與 API 防呆攔截機制，防止 `out-of-bounds` 錯誤。
    * **V2.2 更新：** 導入雙分頁架構。
    """)

# ==========================================
#  分頁 1：量化分析主程式 (Analyzer)
# ==========================================
with tab1:
    st.title("📊 Quant Portfolio Analyzer")
    st.markdown("請在左側輸入資產權重，並確保總和為 100%，然後點擊「開始分析」。")

    # --- 側邊欄：輸入參數 ---
    st.sidebar.header("⚙️ 參數設定 (Parameters)")
    num_assets = st.sidebar.number_input("你的投資組合有幾檔標的？", min_value=1, max_value=10, value=4, step=1)
    st.sidebar.markdown("### 📝 填寫資產與權重")

    default_tickers = ["0050.TW", "BND", "VOO", "GOOG"]
    default_weights = [42, 20, 19, 19]
    tickers_list = []
    weights_list = []

    for i in range(num_assets):
        col1, col2 = st.sidebar.columns([6, 4]) 
        with col1:
            d_ticker = default_tickers[i] if i < len(default_tickers) else ""
            t = st.text_input(f"標的 {i+1}", value=d_ticker, key=f"ticker_{i}")
        with col2:
            d_weight = default_weights[i] if i < len(default_weights) else 0
            w = st.number_input(f"權重(%)", min_value=0, max_value=100, value=d_weight, key=f"weight_{i}")
            
        tickers_list.append(t.strip())
        weights_list.append(w)

    st.sidebar.divider() 
    start_date = st.sidebar.date_input("開始日期", datetime(2021, 1, 1))
    end_date = st.sidebar.date_input("結束日期", datetime.now())
    benchmark_ticker = st.sidebar.text_input("比較基準 (Benchmark)", "0050.TW")

    # --- 核心運算函數 ---
    @st.cache_data
    def get_data(tickers, start, end):
        valid_tickers = [t for t in tickers if t] 
        df = yf.download(valid_tickers, start=start, end=end, auto_adjust=True)
        if 'Close' in df.columns:
            return df['Close']
        return df

    def calculate_metrics(daily_returns, benchmark_returns=None):
        # 【防線 1】如果傳進來的資料是空的，直接回傳 0，避免 iloc[-1] 崩潰
        if len(daily_returns) == 0:
            return 0, 0, 0, 0, 0, 0, 1.0, pd.Series(dtype=float), pd.Series(dtype=float)

        cumulative = (1 + daily_returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        n_years = len(daily_returns) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        volatility = daily_returns.std() * np.sqrt(252)
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        mdd = drawdown.min()
        
        rf = 0.03
        sharpe = (cagr - rf) / volatility if volatility != 0 else 0
        calmar = cagr / abs(mdd) if mdd != 0 else 0
        
        down_capture = 1.0 
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            down_days_mask = benchmark_returns < 0
            if down_days_mask.sum() > 0:
                port_down_ret = (1 + daily_returns[down_days_mask]).prod() - 1
                bench_down_ret = (1 + benchmark_returns[down_days_mask]).prod() - 1
                down_capture = port_down_ret / bench_down_ret if bench_down_ret != 0 else 1.0

        return total_return, cagr, volatility, mdd, sharpe, calmar, down_capture, cumulative, drawdown

    # --- 執行分析按鈕邏輯 ---
    if st.sidebar.button("🚀 開始分析 (Run Analysis)"):
        clean_tickers = []
        clean_weights = []
        for t, w in zip(tickers_list, weights_list):
            if t != "":  
                clean_tickers.append(t)
                clean_weights.append(float(w)/100)
        
        total_weight = sum(clean_weights) * 100
        if abs(total_weight - 100) > 0.1: 
            st.error(f"❌ 錯誤：你的權重總和是 {total_weight:.1f}%，必須剛好等於 100%！")
            st.stop() 

        try:
            with st.spinner('從 Yahoo Finance 撈取數據中...'):
                all_tickers = list(set(clean_tickers + [benchmark_ticker]))
                raw_data = get_data(all_tickers, start_date, end_date)

            # 【防線 2】檢查 Yahoo Finance 是否回傳空資料
            if raw_data.empty:
                st.error("❌ 錯誤：無法從 Yahoo Finance 取得資料。請檢查「股票代號」是否打錯，或「日期區間」內是否有開盤交易。")
                st.stop()

            # 處理掉遺失值
            raw_data = raw_data.dropna(how='all') 

            if isinstance(raw_data, pd.Series):
                raw_data = raw_data.to_frame(name=clean_tickers[0])

            returns = raw_data.pct_change().dropna(how='all')
            
            # 【防線 3】檢查計算報酬率後還有沒有資料
            if returns.empty:
                st.error("❌ 錯誤：計算報酬率後無有效資料。這通常是因為你選擇的「開始日期」太近（例如今天），導致沒有足夠的歷史資料可供計算。請把日期往前調。")
                st.stop()

            # 只保留大家都有資料的欄位 (避免有股票找不到)
            available_tickers = [t for t in clean_tickers if t in returns.columns]
            if len(available_tickers) != len(clean_tickers):
                missing = set(clean_tickers) - set(available_tickers)
                st.warning(f"⚠️ 警告：找不到以下標的的資料：{missing}。可能是代號錯誤。")
                st.stop()

            portfolio_ret = (returns[clean_tickers] * clean_weights).sum(axis=1)
            
            if benchmark_ticker in returns.columns:
                benchmark_ret = returns[benchmark_ticker]
            else:
                st.error(f"❌ 錯誤：找不到比較基準 '{benchmark_ticker}' 的資料。")
                st.stop()

            common_index = portfolio_ret.index.intersection(benchmark_ret.index)
            
            # 【防線 4】檢查對齊日期後還有沒有資料 (例如某檔股票去年才上市)
            if common_index.empty:
                st.error("❌ 錯誤：你輸入的標的與大盤之間，沒有共同的交易日重疊。這可能是因為某檔股票剛上市不久，無法與大盤的長天期資料對齊。")
                st.stop()

            portfolio_ret = portfolio_ret.loc[common_index]
            benchmark_ret = benchmark_ret.loc[common_index]

            p_metrics = calculate_metrics(portfolio_ret, benchmark_ret)
            b_metrics = calculate_metrics(benchmark_ret, benchmark_ret) 

            # --- 顯示結果 ---
            st.subheader("🏆 績效與防禦力總覽")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("總報酬率", f"{p_metrics[0]:.2%}", f"{(p_metrics[0]-b_metrics[0])*100:.2f} p.p.")
            c2.metric("年化報酬 (CAGR)", f"{p_metrics[1]:.2%}", f"{(p_metrics[1]-b_metrics[1])*100:.2f} p.p.")
            c3.metric("夏普比率 (CP值)", f"{p_metrics[4]:.2f}", f"{p_metrics[4]-b_metrics[4]:.2f}")

            st.markdown("<br>", unsafe_allow_html=True) 

            c4, c5, c6, c7 = st.columns(4)
            c4.metric("波動率 (越低越好)", f"{p_metrics[2]:.2%}", f"{(p_metrics[2]-b_metrics[2])*100:.2f} p.p.", delta_color="inverse")
            c5.metric("最大回撤 MDD", f"{p_metrics[3]:.2%}", f"{(p_metrics[3]-b_metrics[3])*100:.2f} p.p.", delta_color="inverse")
            c6.metric("🛡️ 卡瑪比率", f"{p_metrics[5]:.2f}", f"{p_metrics[5]-b_metrics[5]:.2f}")
            c7.metric("🛡️ 下檔捕獲率", f"{p_metrics[6]:.2%}", f"{(p_metrics[6]-b_metrics[6])*100:.2f} p.p.", delta_color="inverse")

            st.divider()

            st.subheader("📈 財富累積曲線 (Wealth Index)")
            fig1 = go.Figure()
            if not p_metrics[7].empty:
                fig1.add_trace(go.Scatter(x=p_metrics[7].index, y=p_metrics[7], mode='lines', name='My Portfolio', line=dict(color='blue', width=2)))
                fig1.add_trace(go.Scatter(x=b_metrics[7].index, y=b_metrics[7], mode='lines', name=benchmark_ticker, line=dict(color='gray', dash='dot')))
            fig1.update_layout(hovermode="x unified")
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("🌊 水下圖 (MDD Analysis)")
            fig2 = go.Figure()
            if not p_metrics[8].empty:
                fig2.add_trace(go.Scatter(x=p_metrics[8].index, y=p_metrics[8], fill='tozeroy', name='My Portfolio', line=dict(color='red')))
            fig2.update_layout(hovermode="x unified", yaxis_tickformat='.0%')
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"發生未預期錯誤：{str(e)}")