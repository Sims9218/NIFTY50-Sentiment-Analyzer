import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Nifty50 Predictor", layout="wide")

def get_latest_prediction():
    if os.path.exists("predictions_log.txt"):
        with open("predictions_log.txt", "r") as f:
            lines = f.readlines()
            if lines:
                return lines[-1].strip()
    return "No prediction data available yet."

st.title("Nifty 50 AI Prediction Dashboard")

st.subheader("Today's Market Forecast")
prediction_text = get_latest_prediction()

if "UP" in prediction_text:
    st.success(f"### Forecast: {prediction_text}")
elif "DOWN" in prediction_text:
    st.error(f"### Forecast: {prediction_text}")
else:
    st.info(f"### Status: {prediction_text}")

st.divider()

try:
    df = pd.read_csv("market_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Latest Sentiment Score", f"{df.iloc[-1]['Sentiment_Score']:.2f}")
        fig_price = px.line(df, x='Date', y='Close', title="Nifty 50 Price Action")
        st.plotly_chart(fig_price, use_container_width=True)

    with col2:
        st.metric("Latest Close", f"₹{df.iloc[-1]['Close']:.2f}")
        fig_sent = px.bar(df, x='Date', y='Sentiment_Score', title="News Sentiment Trend")
        st.plotly_chart(fig_sent, use_container_width=True)

except Exception as e:
    st.warning("Please run the GitHub Action at least once to generate market_data.csv")

with st.expander("View Prediction History"):
    if os.path.exists("predictions_log.txt"):
        with open("predictions_log.txt", "r") as f:
            st.text(f.read())
