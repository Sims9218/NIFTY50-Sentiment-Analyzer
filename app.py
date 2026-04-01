import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Nifty50 HMM Dashboard", layout="wide")

def get_latest():
    if os.path.exists("predictions_log.txt"):
        with open("predictions_log.txt", "r") as f:
            return f.readlines()[-1].strip()
    return "Initializing Model..."

st.title("📈 Nifty 50 Prediction (HMM Edition)")

# Display prediction exactly like before
latest_pred = get_latest()
if "UP" in latest_pred:
    st.success(f"### {latest_pred}")
else:
    st.error(f"### {latest_pred}")

try:
    df = pd.read_csv("market_data.csv")
    st.plotly_chart(px.line(df, x='Date', y='Close', title="Nifty 50 Price History"))
    st.plotly_chart(px.bar(df, x='Date', y='Sentiment_Score', title="FinBERT Sentiment Scores"))
except:
    st.info("Waiting for first automated run to generate charts.")
