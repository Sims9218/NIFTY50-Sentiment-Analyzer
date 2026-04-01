import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Nifty50 Sentiment Dashboard", layout="wide")

st.title("📈 Nifty50 Sentiment & Prediction Dashboard")
st.markdown("This dashboard visualizes market sentiment and AI-driven predictions.")

@st.cache_data
def load_data():
    df = pd.read_csv("market_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    df = load_data()

    latest_row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else latest_row

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Nifty Close", f"₹{latest_row['Close']:.2f}", 
                f"{latest_row['Close'] - prev_row['Close']:.2f}")
    
    sentiment_val = latest_row['Sentiment_Score']
    sentiment_label = "Bullish" if sentiment_val > 0.05 else "Bearish" if sentiment_val < -0.05 else "Neutral"
    col2.metric("Market Sentiment", sentiment_label, f"{sentiment_val:.2f}")
    
    col3.metric("System Status", "Live", "Auto-Retraining On")

    st.subheader("Market Trends vs. Sentiment")
    
    fig_price = px.line(df, x='Date', y='Close', title="Nifty 50 Closing Price")
    st.plotly_chart(fig_price, use_container_width=True)

    fig_sent = px.bar(df, x='Date', y='Sentiment_Score', 
                      title="Daily Sentiment Index (FinBERT)",
                      color='Sentiment_Score', 
                      color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_sent, use_container_width=True)

    if st.checkbox("Show Raw Data"):
        st.write(df.tail(10))

except Exception as e:
    st.error(f"Waiting for data... Ensure market_data.csv exists. Error: {e}")
