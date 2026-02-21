import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- 網頁設定 ---
st.set_page_config(page_title="雙博士投資組合分析儀 V3.2", layout="wide")

# --- 建立三分頁 (Tabs) ---
# 建立四個主要分頁 (為了不影響你原本的變數，我們把新分頁叫 tab4 並安插在第三個位置)
tab1, tab3, tab4, tab2 = st.tabs([
    "📊 量化分析 (Analyzer)", 
    "⚔️ ETF 擂台 (Compare)", 
    "🌱 定期定額推演 (DCA)", 
    "ℹ️ 系統資訊 (About)"
])

# ==========================================
#  分頁 4 (UI 排序第 3)：定期定額複利推演 (DCA)
# ==========================================
with tab4:
    st.header("🌱 巴菲特時間魔法：定期定額財富推演")
    st.info("💡 這裡的推演模擬了真實世界中投資人「初始一筆資金 + 每月持續投入」的資產增長軌跡。")

    col1, col2, col3 = st.columns(3)
    with col1:
        initial_capital = st.number_input("初始單筆本金 (元)", min_value=0, value=100000, step=10000)
    with col2:
        monthly_investment = st.number_input("每月定期定額投入 (元)", min_value=0, value=10000, step=1000)
    with col3:
        expected_cagr = st.number_input("預期年化報酬率 (%)", min_value=0.0, value=7.0, step=0.5)

    invest_years = st.slider("預計投資年限 (年)", min_value=1, max_value=50, value=20)

    # 複利計算邏輯 (按月滾動)
    months = invest_years * 12
    monthly_rate = (expected_cagr / 100) / 12
    
    current_value = initial_capital
    total_principal = initial_capital
    
    values_list = [current_value]
    principals_list = [total_principal]
    
    for i in range(months):
        current_value = current_value * (1 + monthly_rate) + monthly_investment
        total_principal += monthly_investment
        
        # 每年年底記錄一次數據畫圖用
        if (i + 1) % 12 == 0:
            values_list.append(current_value)
            principals_list.append(total_principal)

    import plotly.graph_objects as go
    years_x = list(range(invest_years + 1))
    
    fig_fv = go.Figure()
    fig_fv.add_trace(go.Scatter(x=years_x, y=values_list, mode='lines', name='總資產價值', 
                                line=dict(color='green', width=3), fill='tozeroy', hovertemplate='$%{y:,.0f}'))
    fig_fv.add_trace(go.Scatter(x=years_x, y=principals_list, mode='lines', name='累積投入本金', 
                                line=dict(color='gray', dash='dash', width=2), hovertemplate='$%{y:,.0f}'))
    
    fig_fv.update_layout(title="定期定額複利成長曲線", xaxis_title="投資年數", yaxis_title="累積金額", hovermode="x unified")
    st.plotly_chart(fig_fv, use_container_width=True)

    # 總結數據卡片
    m1, m2, m3 = st.columns(3)
    m1.metric("總投入本金", f"${total_principal:,.0f}")
    m2.metric("最終總資產", f"${current_value:,.0f}")
    m3.metric("純複利獲利", f"${(current_value - total_principal):,.0f}")


# ==========================================
#  分頁 3：系統資訊 (About)
# ==========================================
with tab2:
    st.header("ℹ️ 關於本系統 (About)")
    st.markdown(f"""
    **雙博士投資組合分析儀 (Quant Portfolio Analyzer)** 是一個專為量化投資人打造的專業級回測與風險評估工具。
    
    ### 👨‍💻 開發團隊 (Credits)
    * **系統架構與主開發者：** [Alvin Zhang (BA, History, NTU)](https://www.linkedin.com/in/kun-jie-zhang-376902284/) (圖書資訊系研究生)
    * **AI 協同開發顧問：** Google Gemini (雙博士理財與資工顧問)
    * **核心運算引擎：** Python, Streamlit, Pandas, yfinance, Plotly
    
    ---
    ### 🔄 版本更新紀錄 (Changelog)
    * **V3.2 (Hotfix)：** 修復跨國資產休市不對齊錯誤，導入 ffill() 機制。
    * **V3.1 更新：** 指標改為 % 顯示，新增 1 年期滾動報酬與勝率分析。
    """)

# ==========================================
#  分頁 2：ETF 擂台 (Compare)
# ==========================================
with tab3:
    st.title("⚔️ ETF 終極擂台")
    st.markdown("輸入兩檔 ETF 進行一對一單挑。系統將自動計算歷史績效，並嘗試抓取資產規模等基本面數據。")
    
    col_a, col_b = st.columns(2)
    with col_a:
        etf_a = st.text_input("輸入第一檔 ETF 代號 (例如 VOO)", value="VOO").strip().upper()
    with col_b:
        etf_b = st.text_input("輸入第二檔 ETF 代號 (例如 SPY)", value="SPY").strip().upper()
        
    c_start_date = st.date_input("比較起始日期", datetime(2020, 1, 1), key="compare_start")
    
    def calc_single_asset_metrics(daily_returns):
        if len(daily_returns) == 0:
            return 0, 0, 0, 0, 0
        cumulative = (1 + daily_returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        n_years = len(daily_returns) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        volatility = daily_returns.std() * np.sqrt(252)
        running_max = cumulative.cummax()
        mdd = ((cumulative - running_max) / running_max).min()
        sharpe = (cagr - 0.03) / volatility if volatility != 0 else 0
        return total_return, cagr, volatility, mdd, sharpe
    
    def format_currency(num):
        if num is None or pd.isna(num): return "N/A"
        if num >= 1e9: return f"{num/1e9:.2f} B (十億)"
        if num >= 1e6: return f"{num/1e6:.2f} M (百萬)"
        return f"{num:,.0f}"

    if st.button("🔥 開始對決 (Fight!)"):
        if etf_a and etf_b:
            try:
                with st.spinner('正在同步爬取歷史價格與基本面資料...'):
                    df_dl = yf.download([etf_a, etf_b], start=c_start_date, auto_adjust=True, progress=False)
                    if 'Close' in df_dl.columns:
                        df_compare = df_dl['Close']
                    else:
                        df_compare = df_dl

                    # 【V3.2 修復】使用 ffill 填補休市價格，避免跨國 ETF 比較時資料遺失
                    df_compare = df_compare.ffill().dropna(how='all') 
                    
                    if df_compare.empty:
                        st.error("❌ 無法取得共同的歷史交易資料，請檢查代號是否正確。")
                        st.stop()
                        
                    ret_a = df_compare[etf_a].pct_change().fillna(0)
                    ret_b = df_compare[etf_b].pct_change().fillna(0)
                    
                    metrics_a = calc_single_asset_metrics(ret_a)
                    metrics_b = calc_single_asset_metrics(ret_b)
                    
                    info_a = yf.Ticker(etf_a).info
                    info_b = yf.Ticker(etf_b).info
                    
                st.divider()
                st.subheader("📊 基本面資料比拚")
                st.caption("⚠️ 註：Yahoo Finance 對非美股 (如台股) 的基本面資料覆蓋率較低，若顯示 N/A 代表官方 API 查無資料。")
                
                comp_data = {
                    "評估項目": ["總資產規模 (AUM)", "殖利率 (Yield)", "52週最高價", "52週最低價"],
                    f"🔵 {etf_a}": [
                        format_currency(info_a.get('totalAssets')), 
                        f"{info_a.get('yield', 0)*100:.2f}%" if info_a.get('yield') else "N/A",
                        info_a.get('fiftyTwoWeekHigh', 'N/A'),
                        info_a.get('fiftyTwoWeekLow', 'N/A')
                    ],
                    f"🔴 {etf_b}": [
                        format_currency(info_b.get('totalAssets')), 
                        f"{info_b.get('yield', 0)*100:.2f}%" if info_b.get('yield') else "N/A",
                        info_b.get('fiftyTwoWeekHigh', 'N/A'),
                        info_b.get('fiftyTwoWeekLow', 'N/A')
                    ]
                }
                st.table(pd.DataFrame(comp_data).set_index("評估項目"))
                
                st.subheader("🕸️ 戰力雷達圖 (Radar Chart)")
                st.info("💡 白話解釋：涵蓋面積越大的標的，代表其綜合戰鬥力（報酬高、波動低、抗跌能力強）越優秀！")
                
                categories = ['年化報酬(CAGR)', '夏普比率(CP值)', '抗跌力(反轉MDD)', '穩定度(反轉波動率)']
                
                val_a = [metrics_a[1]*100, metrics_a[4]*10, (1+metrics_a[3])*100, (1-metrics_a[2])*100]
                val_b = [metrics_b[1]*100, metrics_b[4]*10, (1+metrics_b[3])*100, (1-metrics_b[2])*100]
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=val_a, theta=categories, fill='toself', name=etf_a, line_color='blue'))
                fig_radar.add_trace(go.Scatterpolar(r=val_b, theta=categories, fill='toself', name=etf_b, line_color='red'))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), showlegend=True, margin=dict(t=20, b=20))
                
                col_r1, col_radar, col_r2 = st.columns([1, 2, 1])
                with col_radar:
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
            except Exception as e:
                st.error(f"比對過程中發生錯誤：{str(e)}")

# ==========================================
#  分頁 1：量化分析主程式 (Analyzer)
# ==========================================
with tab1:
    st.title("📊 Quant Portfolio Analyzer")
    st.markdown("請在左側輸入資產權重，並確保總和為 100%，然後點擊「開始分析」。")

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
        tickers_list.append(t)
        weights_list.append(w)

    st.sidebar.divider() 
    start_date = st.sidebar.date_input("開始日期", datetime(2015, 1, 1))
    end_date = st.sidebar.date_input("結束日期", datetime.now())

    st.sidebar.markdown("### 🔄 進階分析設定")
    rolling_years = st.sidebar.selectbox(
        "選擇滾動分析週期 (Rolling Period)：", 
        options=[1, 3, 5], 
        format_func=lambda x: f"{x} 年期"
    )
    rolling_days = rolling_years * 252 # 每年約 252 個交易日
    st.sidebar.divider()
    
    st.sidebar.markdown("### 🎯 比較基準設定")
    BENCHMARKS = {
        "🇹🇼 台灣 50 大盤 (0050.TW)": "0050.TW",
        "🇺🇸 S&P 500 美股大盤 (SPY)": "SPY",
        "🇺🇸 S&P 500 美股大盤 (VOO)": "VOO",
        "🇺🇸 Nasdaq 科技股 (QQQ)": "QQQ",
        "🌍 全球股票大盤 (VT)": "VT",
        "🇺🇸 美國整體股市 (VTI)": "VTI",
        "🏦 美國整體債券 (BND)": "BND",
        "✏️ 自訂輸入 (Custom)": "CUSTOM"
    }
    
    selected_bench_name = st.sidebar.selectbox("選擇你要挑戰的大盤：", list(BENCHMARKS.keys()))
    
    if BENCHMARKS[selected_bench_name] == "CUSTOM":
        raw_benchmark = st.sidebar.text_input("請輸入自訂代號 (例如 AAPL)", "0050.TW").strip().upper()
    else:
        raw_benchmark = BENCHMARKS[selected_bench_name]

    st.sidebar.markdown("<br><br>", unsafe_allow_html=True) 
    st.sidebar.info(f"""
    👨‍💻 **Developed by:** [Alvin Zhang (BA, History, NTU)](https://www.linkedin.com/in/kun-jie-zhang-376902284/)  
    🤖 **Co-Pilot:** Gemini AI
    """)

    @st.cache_data
    def get_data(tickers, start, end):
        valid_tickers = [t for t in tickers if t] 
        df = yf.download(valid_tickers, start=start, end=end, auto_adjust=True, progress=False)
        if 'Close' in df.columns:
            return df['Close']
        return df

    def calculate_metrics(daily_returns, benchmark_returns=None):
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

    if st.sidebar.button("🚀 開始分析 (Run Analysis)"):
        clean_tickers = []
        clean_weights = []
        for t, w in zip(tickers_list, weights_list):
            t_clean = t.strip().upper()
            if t_clean != "":  
                clean_tickers.append(t_clean)
                clean_weights.append(float(w)/100)
                
        benchmark_ticker = raw_benchmark.strip().upper()
        total_weight = sum(clean_weights) * 100
        
        if abs(total_weight - 100) > 0.1: 
            st.error(f"❌ 錯誤：你的權重總和是 {total_weight:.1f}%，必須剛好等於 100%！")
            st.stop() 

        try:
            with st.spinner('從 Yahoo Finance 撈取數據中...'):
                all_tickers = list(set(clean_tickers + [benchmark_ticker]))
                raw_data = get_data(all_tickers, start_date, end_date)

            if isinstance(raw_data, pd.Series):
                raw_data = raw_data.to_frame(name=all_tickers[0])

            downloaded_columns = raw_data.columns.tolist()
            if benchmark_ticker not in downloaded_columns or raw_data[benchmark_ticker].dropna().empty:
                st.error(f"❌ 代號錯誤：找不到比較基準 **'{benchmark_ticker}'** 的資料！請確認代號是否輸入正確。")
                st.stop()

            invalid_tickers = []
            for t in clean_tickers:
                if t not in downloaded_columns or raw_data[t].dropna().empty:
                    invalid_tickers.append(t)
            if invalid_tickers:
                st.error(f"❌ 代號錯誤：找不到以下標的 **{invalid_tickers}** 的資料！請確認代號是否輸入正確。")
                st.stop()

            # 【V3.2 修復】使用 ffill 延續前一天價格，解決台股美股休假日不同的問題
            raw_data = raw_data.ffill().dropna(how='all') 
            returns = raw_data.pct_change().fillna(0) # 休市日報酬率設為 0%
            
            if returns.empty:
                st.error("❌ 錯誤：計算報酬率後無有效資料。這通常是因為你選擇的日期區間太短或遇到連續休市。")
                st.stop()

            portfolio_ret = (returns[clean_tickers] * clean_weights).sum(axis=1)
            benchmark_ret = returns[benchmark_ticker]
            common_index = portfolio_ret.index.intersection(benchmark_ret.index)
            
            if common_index.empty:
                st.error("❌ 錯誤：你輸入的標的與大盤之間，沒有共同的交易日重疊。")
                st.stop()

            portfolio_ret = portfolio_ret.loc[common_index]
            benchmark_ret = benchmark_ret.loc[common_index]
            p_metrics = calculate_metrics(portfolio_ret, benchmark_ret)
            b_metrics = calculate_metrics(benchmark_ret, benchmark_ret) 

            # --- 顯示結果 UI (改為 % 顯示) ---
            st.subheader("🏆 績效與防禦力總覽")
            
            st.markdown(f"**🆚 比較基準：** 以下數字下方的小字，皆為與 **{selected_bench_name}** 比較的差距。")
            
            with st.expander("💡 點我查看：各項專業指標白話解釋"):
                st.markdown("""
                * **📊 夏普比率 (Sharpe Ratio) - 投資的「CP 值」：** 衡量你「承受每1%的波動，能換來多少超額報酬」。通常 > 1 代表優秀。
                * **🛡️ 卡瑪比率 (Calmar Ratio) - 投資的「抗跌效能」：** 衡量你「承受每1%的最大虧損，能換來多少年化報酬」。> 1 視為神級抗跌策略。
                * **🧲 下檔捕獲率 (Downside Capture) - 投資的「防禦裝甲」：** 衡量「大盤下跌時，你跟著跌了多少」。越低越好。*(例如：80% 代表大盤跌 10 元，你只跌 8 元)*
                """)
            
            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("總報酬率", f"{p_metrics[0]:.2%}", f"{(p_metrics[0]-b_metrics[0])*100:.2f}%")
            c2.metric("年化報酬 (CAGR)", f"{p_metrics[1]:.2%}", f"{(p_metrics[1]-b_metrics[1])*100:.2f}%")
            c3.metric("夏普比率 (CP值)", f"{p_metrics[4]:.2f}", f"{p_metrics[4]-b_metrics[4]:.2f}")
            
            st.markdown("<br>", unsafe_allow_html=True) 
            c4, c5, c6, c7 = st.columns(4)
            c4.metric("波動率 (越低越好)", f"{p_metrics[2]:.2%}", f"{(p_metrics[2]-b_metrics[2])*100:.2f}%", delta_color="inverse")
            c5.metric("最大回撤 MDD", f"{p_metrics[3]:.2%}", f"{(p_metrics[3]-b_metrics[3])*100:.2f}%", delta_color="inverse")
            c6.metric("🛡️ 卡瑪比率", f"{p_metrics[5]:.2f}", f"{p_metrics[5]-b_metrics[5]:.2f}")
            c7.metric("🛡️ 下檔捕獲率", f"{p_metrics[6]:.2%}", f"{(p_metrics[6]-b_metrics[6])*100:.2f}%", delta_color="inverse")
            st.divider()

           # --- 動態滾動報酬與勝率分析 ---
            st.subheader(f"🔄 歷史勝率與滾動報酬 ({rolling_years}-Year Rolling Returns)")
            st.info(f"**💡 系統解讀：** 假設你在過去任何一個交易日進場，並且**堅持持有 {rolling_years} 年**，以下是你的真實勝率與報酬分佈。週期越長，通常勝率會越高且越穩定。")

            if len(p_metrics[7]) > rolling_days:
                # 根據使用者選擇的年份，動態切換計算天數 (rolling_days)
                port_roll = (p_metrics[7] / p_metrics[7].shift(rolling_days)) - 1
                bench_roll = (b_metrics[7] / b_metrics[7].shift(rolling_days)) - 1
                
                roll_df = pd.DataFrame({'port': port_roll, 'bench': bench_roll}).dropna()

                if not roll_df.empty:
                    win_rate = (roll_df['port'] > 0).mean()
                    beat_market_rate = (roll_df['port'] > roll_df['bench']).mean()

                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric(f"持有 {rolling_years} 年賺錢機率 (勝率)", f"{win_rate:.2%}")
                    rc2.metric(f"持有 {rolling_years} 年打敗大盤機率", f"{beat_market_rate:.2%}")
                    rc3.metric(f"平均 {rolling_years} 年期報酬率", f"{roll_df['port'].mean():.2%}")

                    fig_roll = go.Figure()
                    fig_roll.add_trace(go.Scatter(x=roll_df.index, y=roll_df['port'], mode='lines', name=f'我的組合 ({rolling_years}年期)', line=dict(color='purple'), hovertemplate='%{y:.2%}'))
                    fig_roll.add_trace(go.Scatter(x=roll_df.index, y=roll_df['bench'], mode='lines', name=f'{benchmark_ticker} ({rolling_years}年期)', line=dict(color='gray', dash='dot'), hovertemplate='%{y:.2%}'))
                    fig_roll.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="0% (損益兩平線)", annotation_position="bottom right")
                    
                    fig_roll.update_layout(hovermode="x unified", yaxis_tickformat='.0%')
                    st.plotly_chart(fig_roll, use_container_width=True)
                else:
                    st.warning("⚠️ 對齊資料後無有效區間可供計算。")
            else:
                st.warning(f"⚠️ 你的資料區間不足 {rolling_years} 年 (少於 {rolling_days} 個交易日)，無法計算滾動報酬。請將左側的「開始日期」往前調！")
            
            st.divider()

            # --- 甜甜圈圖 ---
            st.subheader("🍩 資產配置權重 (Asset Allocation)")
            fig_pie = go.Figure(data=[go.Pie(labels=clean_tickers, values=clean_weights, hole=0.4, textinfo='label+percent', insidetextorientation='radial')])
            fig_pie.update_layout(margin=dict(t=20, b=20, l=0, r=0), height=350)
            col_space1, col_pie, col_space2 = st.columns([1, 2, 1])
            with col_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
            st.divider()

            # --- 走勢圖與水下圖 ---
            st.subheader("📈 財富累積曲線 (Wealth Index)")
            st.info("**💡 白話解釋：** 假設你在起點投入了 **1 元**，這條線代表你總資產的成長變化。大盤的虛線讓你一眼看出有沒有跑贏大盤。")
            fig1 = go.Figure()
            if not p_metrics[7].empty:
                fig1.add_trace(go.Scatter(x=p_metrics[7].index, y=p_metrics[7], mode='lines', name='我的投資組合', line=dict(color='blue', width=2), hovertemplate='%{y:.2f} 倍'))
                fig1.add_trace(go.Scatter(x=b_metrics[7].index, y=b_metrics[7], mode='lines', name=f'大盤 ({benchmark_ticker})', line=dict(color='gray', dash='dot'), hovertemplate='%{y:.2f} 倍'))
            fig1.update_layout(hovermode="x unified")
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("🌊 水下圖 (Underwater Chart / Drawdown)")
            st.info("**💡 白話解釋（套牢痛苦指數）：** 專門衡量股災時的「虧損痛感」。**0%** 代表沒有虧損。跌到 **-20%**，代表資產從最高點縮水了 20%。")
            fig2 = go.Figure()
            if not p_metrics[8].empty:
                fig2.add_trace(go.Scatter(x=p_metrics[8].index, y=p_metrics[8], fill='tozeroy', name='我的投資組合', line=dict(color='red'), hovertemplate='回撤跌幅: %{y:.2%}'))
            fig2.update_layout(hovermode="x unified", yaxis_tickformat='.0%')
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"發生未預期錯誤：{str(e)}")