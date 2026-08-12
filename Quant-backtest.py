import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="QuantBacktest Analytics",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# BACKTESTING ENGINE
# ==========================================
class BacktestEngine:
    def __init__(self, ticker: str, start_date: str, end_date: str, initial_capital: float = 100000.0):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = pd.DataFrame()

    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance."""
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if df.empty:
            raise ValueError(f"No data retrieved for ticker '{self.ticker}'.")
        
        # Handle MultiIndex columns if returned by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(self.ticker, level=1, axis=1)

        self.data = df[['Close']].copy()
        return self.data

    def run_sma_crossover(self, short_window: int, long_window: int):
        """Executes a Dual Moving Average Crossover Strategy."""
        df = self.data.copy()

        # Calculate Moving Averages
        df['Short_SMA'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
        df['Long_SMA'] = df['Close'].rolling(window=long_window, min_periods=1).mean()

        # Generate Signals: 1 (Long), 0 (Cash)
        df['Signal'] = np.where(df['Short_SMA'] > df['Long_SMA'], 1.0, 0.0)

        # Calculate Positions (Trades occur on signal changes)
        df['Position'] = df['Signal'].shift(1).fillna(0)

        # Returns Calculation
        df['Market_Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Market_Returns'] * df['Position']

        # Cumulative Returns & Equity Curves
        df['Cumulative_Market'] = (1 + df['Market_Returns'].fillna(0)).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Returns'].fillna(0)).cumprod()
        df['Portfolio_Value'] = self.initial_capital * df['Cumulative_Strategy']

        self.data = df
        return df

    def calculate_metrics(self) -> dict:
        """Computes key quantitative performance metrics."""
        df = self.data.dropna().copy()
        if df.empty:
            return {}

        total_days = (df.index[-1] - df.index[0]).days
        years = max(total_days / 365.25, 0.001)

        # Final Return & CAGR
        final_value = df['Portfolio_Value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        cagr = ((final_value / self.initial_capital) ** (1 / years)) - 1

        # Annualized Volatility
        daily_std = df['Strategy_Returns'].std()
        annualized_vol = daily_std * np.sqrt(252)

        # Sharpe Ratio (Assuming 5% Risk-Free Rate)
        risk_free_rate = 0.05
        excess_returns = df['Strategy_Returns'] - (risk_free_rate / 252)
        sharpe_ratio = (excess_returns.mean() / daily_std * np.sqrt(252)) if daily_std != 0 else 0.0

        # Maximum Drawdown
        rolling_max = df['Portfolio_Value'].cummax()
        drawdown = (df['Portfolio_Value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Win Rate Calculations
        trade_signals = df['Position'].diff()
        buy_trades = df[trade_signals == 1]
        sell_trades = df[trade_signals == -1]

        trades = []
        for i in range(min(len(buy_trades), len(sell_trades))):
            entry_price = buy_trades['Close'].iloc[i]
            exit_price = sell_trades['Close'].iloc[i]
            trade_return = (exit_price - entry_price) / entry_price
            trades.append(trade_return)

        win_rate = (sum(1 for t in trades if t > 0) / len(trades)) if len(trades) > 0 else 0.0

        return {
            "Final Portfolio Value ($)": round(final_value, 2),
            "Total Return (%)": round(total_return * 100, 2),
            "CAGR (%)": round(cagr * 100, 2),
            "Annualized Volatility (%)": round(annualized_vol * 100, 2),
            "Sharpe Ratio": round(sharpe_ratio, 2),
            "Max Drawdown (%)": round(max_drawdown * 100, 2),
            "Total Trades Executed": len(trades),
            "Win Rate (%)": round(win_rate * 100, 2)
        }

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================
st.title("📈 Quantitative Backtesting & Strategy Analytics Engine")
st.markdown("A professional backtesting terminal to design, evaluate, and analyze technical trading strategies.")

# Sidebar Configuration
st.sidebar.header("Strategy Settings")
ticker = st.sidebar.text_input("Ticker Symbol (Yahoo Finance)", value="RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2026-01-01"))
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000, step=5000)

st.sidebar.subheader("SMA Crossover Parameters")
short_sma = st.sidebar.slider("Fast Moving Average (Days)", min_value=5, max_value=50, value=20)
long_sma = st.sidebar.slider("Slow Moving Average (Days)", min_value=20, max_value=200, value=50)

run_button = st.sidebar.button("Run Backtest", type="primary")

if run_button:
    if short_sma >= long_sma:
        st.error("Error: Fast Moving Average window must be strictly less than Slow Moving Average window.")
    else:
        with st.spinner("Fetching data and computing analytics..."):
            try:
                # Initialize and run backtest engine
                engine = BacktestEngine(ticker, str(start_date), str(end_date), initial_capital)
                engine.fetch_data()
                df_results = engine.run_sma_crossover(short_sma, long_sma)
                metrics = engine.calculate_metrics()

                # --- KPI Metrics Display ---
                st.subheader("📊 Strategy Performance Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Final Capital", f"${metrics['Final Portfolio Value ($)']:,}", f"{metrics['Total Return (%)']}% Total")
                col2.metric("CAGR", f"{metrics['CAGR (%)']}%")
                col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']}")
                col4.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']}%")

                col5, col6, col7, col8 = st.columns(4)
                col5.metric("Volatility (Annual)", f"{metrics['Annualized Volatility (%)']}%")
                col6.metric("Total Trades", f"{metrics['Total Trades Executed']}")
                col7.metric("Win Rate", f"{metrics['Win Rate (%)']}%")
                col8.metric("Initial Investment", f"${initial_capital:,}")

                st.markdown("---")

                # --- Visualizations ---
                st.subheader("📉 Performance & Signals Charts")
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

                # Plot 1: Price and Moving Averages
                ax1.plot(df_results.index, df_results['Close'], label='Asset Price', color='black', alpha=0.6)
                ax1.plot(df_results.index, df_results['Short_SMA'], label=f'{short_sma}-Day SMA', color='blue', linestyle='--')
                ax1.plot(df_results.index, df_results['Long_SMA'], label=f'{long_sma}-Day SMA', color='orange', linestyle='--')

                # Plot Buy/Sell signals
                buy_signals = df_results[df_results['Position'].diff() == 1]
                sell_signals = df_results[df_results['Position'].diff() == -1]
                ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal', zorder=5)
                ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell Signal', zorder=5)

                ax1.set_title(f"{ticker} Price & Moving Average Crossover Signals")
                ax1.set_ylabel("Price")
                ax1.legend(loc="upper left")
                ax1.grid(True, alpha=0.3)

                # Plot 2: Portfolio Equity Curve vs Market
                ax1_val = df_results['Portfolio_Value']
                benchmark_val = initial_capital * df_results['Cumulative_Market']
                
                ax2.plot(df_results.index, ax1_val, label='Strategy Portfolio', color='green', linewidth=2)
                ax2.plot(df_results.index, benchmark_val, label='Buy & Hold Benchmark', color='gray', linestyle=':')
                ax2.set_title("Equity Curve Comparison ($)")
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Portfolio Value ($)")
                ax2.legend(loc="upper left")
                ax2.grid(True, alpha=0.3)

                st.pyplot(fig)

                # --- Raw Data Logs ---
                with st.expander("🔍 View Detailed Dataset Logs"):
                    st.dataframe(df_results[['Close', 'Short_SMA', 'Long_SMA', 'Position', 'Strategy_Returns', 'Portfolio_Value']].tail(100))

            except Exception as e:
                st.error(f"Execution failed: {str(e)}")
else:
    st.info("👈 Set your strategy parameters in the sidebar and click **Run Backtest** to launch the analysis.")