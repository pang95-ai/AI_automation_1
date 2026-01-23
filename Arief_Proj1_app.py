# app.py

import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# 1. Page config
# -----------------------------

st.set_page_config(
    page_title="Arief Stock / ETF Dashboard",
    layout="wide",)

st.title("📈 Stock & ETF Dashboard")


# -----------------------------
# 2. Sidebar controls
# -----------------------------

st.sidebar.header("Controls")

# Ticker input
ticker = st.sidebar.text_input("Ticker (e.g., AAPL, SPY, QQQ)", value="AAPL")

# Date range
default_start = dt.date.today() - dt.timedelta(days=365)
default_end = dt.date.today()

start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=default_end)

# Metrics selection
metrics = st.sidebar.multiselect(
    "Metrics to show",
    options=["Close", "Open", "High", "Low", "Adj Close", "Volume"],
    default=["Close", "Volume"],
)

# Moving average windows
ma_short = st.sidebar.number_input("Short MA window (days)", min_value=2, max_value=100, value=20)
ma_long = st.sidebar.number_input("Long MA window (days)", min_value=5, max_value=200, value=50)

# Button to fetch data
fetch_button = st.sidebar.button("Fetch data")


# -----------------------------
# 3. Helper: download data
# -----------------------------

@st.cache_data
def load_data(ticker_symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(ticker_symbol, start=start, end=end)
    return df

# -----------------------------
# 4. Main logic
# -----------------------------

if fetch_button:
    if not ticker:
        st.error("Please enter a ticker symbol.")
        st.stop()

    data = load_data(ticker, start_date, end_date)

    if data.empty:
        st.error("No data returned. Check ticker or date range.")
        st.stop()

    # Reset index so Date becomes a column for Plotly
    data = data.reset_index()

# -----------------------------
# 5. Compute rolling averages
# -----------------------------

data[f"MA_{ma_short}"] = data["Close"].rolling(window=ma_short).mean()
data[f"MA_{ma_long}"] = data["Close"].rolling(window=ma_long).mean()

st.subheader(f"Raw data for {ticker}")
st.dataframe(data.tail())

# -----------------------------
# 6. Download CSV (bonus)
# -----------------------------
  
csv = data.to_csv(index=False).encode("utf-8")
st.download_button(label="Download data as CSV",
data=csv,file_name=f"{ticker}_{start_date}_{end_date}.csv",
        mime="text/csv",)

# -----------------------------
# 7. Price chart with Moving Averages
# -----------------------------
    
if "Close" in metrics:
    st.subheader("Price & Moving Averages")

    fig_price = px.line(data,x="Date",
    y="Close",title=f"{ticker} Closing Price",)

# Add moving averages
    fig_price.add_scatter(
            x=data["Date"],
            y=data[f"MA_{ma_short}"],
            mode="lines",
            name=f"MA {ma_short}",
        )
    fig_price.add_scatter(
            x=data["Date"],
            y=data[f"MA_{ma_long}"],
            mode="lines",
            name=f"MA {ma_long}",)

    fig_price.update_layout(xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_price, use_container_width=True)

    # Bonus: download chart as HTML
    price_html = fig_price.to_html(include_plotlyjs="cdn")
    st.download_button(
            label="Download price chart as HTML",
            data=price_html,
            file_name=f"{ticker}_price_chart.html",
            mime="text/html",)

# -----------------------------
# 8. Volume chart
# -----------------------------

    if "Volume" in metrics and "Volume" in data.columns:
        st.subheader("Volume")

        fig_vol = px.bar(
            data,
            x="Date",
            y="Volume",
            title=f"{ticker} Volume",)

        fig_vol.update_layout(xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(fig_vol, use_container_width=True)

        vol_html = fig_vol.to_html(include_plotlyjs="cdn")
        st.download_button(
            label="Download volume chart as HTML",
            data=vol_html,
            file_name=f"{ticker}_volume_chart.html",
            mime="text/html",)

# -----------------------------
# 9. Other selected metrics
# -----------------------------

    other_metrics = [m for m in metrics if m not in ["Close", "Volume"]]
    if other_metrics:
        st.subheader("Other metrics")

        for m in other_metrics:
            if m in data.columns:
                fig = px.line(
                    data,
                    x="Date",
                    y=m,
                    title=f"{ticker} {m}",)
                fig.update_layout(xaxis_title="Date", yaxis_title=m)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"{m} not available in data.")

else:
    st.info("Set your options in the sidebar and click **Fetch data** to begin.")