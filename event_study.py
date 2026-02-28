import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

class EventStudyAnalyzer:
    def __init__(self, ticker, benchmark="SPY"):
        self.ticker = ticker
        self.benchmark = benchmark
        self.data = None

    def fetch_data(self, event_date, days_before=150, days_after=30):
        # 1. 設定抓取的時間區間
        start_date = (pd.to_datetime(event_date) - pd.Timedelta(days=days_before + 50)).strftime('%Y-%m-%d')
        end_date = (pd.to_datetime(event_date) + pd.Timedelta(days=days_after + 5)).strftime('%Y-%m-%d')
        
        # 2. 下載資料，加入 auto_adjust=True 確保與你的 app.py 邏輯一致
        df_stock_raw = yf.download(self.ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        df_bench_raw = yf.download(self.benchmark, start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        # 3. 檢查欄位名稱（預防萬一，如果沒有 Close 就抓最後一欄）
        def get_price_column(df):
            if 'Close' in df.columns:
                # ✅ 關鍵修正：加上 .squeeze()，把二維表格強制壓扁成一維數列
                return df['Close'].squeeze()
            # ✅ 關鍵修正：加上 .squeeze()
            return df.iloc[:, -1].squeeze()

        s_price = get_price_column(df_stock_raw)
        m_price = get_price_column(df_bench_raw)
        
        # 4. 合併並計算報酬率
        self.data = pd.DataFrame({
            'stock': s_price.pct_change(),
            'market': m_price.pct_change()
        }).dropna()
        
        return self.data

    def run_analysis(self, event_date, window_size=10):
        # 1. 定義窗口
        event_dt = pd.to_datetime(event_date)
        # 估計窗口：事件前 120 天到 前 10 天
        estimation_period = self.data[self.data.index < event_dt].iloc[-120:-10]
        # 事件窗口：事件前後
        event_period = self.data[(self.data.index >= event_dt - pd.Timedelta(days=2)) & 
                                (self.data.index <= event_dt + pd.Timedelta(days=window_size))]

        # 2. 線性回歸 (Market Model)
        # $R_i = \alpha + \beta R_m + \epsilon$
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            estimation_period['market'], estimation_period['stock']
        )

        # 3. 計算異常報酬 (AR) 與 累積異常報酬 (CAR)
        event_period = event_period.copy()
        event_period['expected_return'] = intercept + slope * event_period['market']
        event_period['AR'] = event_period['stock'] - event_period['expected_return']
        event_period['CAR'] = event_period['AR'].cumsum()
        
        return event_period, slope

    def plot_car(self, event_results):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=event_results.index, y=event_results['CAR'] * 100,
                                 mode='lines+markers', name='累積異常報酬 (CAR %)',
                                 line=dict(color='firebrick', width=3)))
        fig.update_layout(title=f"{self.ticker} 事件分析: 累積異常報酬率",
                          xaxis_title="日期", yaxis_title="CAR (%)",
                          hovermode="x unified")
        return fig