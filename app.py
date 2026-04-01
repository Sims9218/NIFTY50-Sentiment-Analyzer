import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Nifty50 HMM Predictor", layout="wide")

def get_latest_pred():
    if os.path.exists("predictions_log.txt"):
        with open("predictions_log.txt", "r") as f:
            return f.readlines()[-1].strip()
    return "Awaiting Model Run..."

st.title("📊 Nifty 50 Hidden Markov Model Dashboard")

pred = get_latest_pred()
if "UP" in pred:
    st.success(f"### Next Day Forecast: {pred}")
else:
    st.error(f"### Next Day Forecast: {pred}")

try:
    df = pd.read_csv("market_data.csv")
    
    st.subheader("Market Regimes (HMM States)")
    fig = px.scatter(df, x='Date', y='Close', color='Sentiment_Score',
                     size=df['Price_Change'].abs(), title="Closing Price vs Sentiment Intensity")
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.line(df, x='Date', y='Close', title="Price Trend"))
    with col2:
        st.plotly_chart(px.bar(df, x='Date', y='Sentiment_Score', title="FinBERT Sentiment History"))

except Exception as e:
    st.info("Upload market_data.csv to begin visualization.")
