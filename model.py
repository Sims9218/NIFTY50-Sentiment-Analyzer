import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import joblib
import os

DATA_PATH = "market_data.csv"
MODEL_PATH = "nifty_hmm.pkl"

def train_hmm(df):
    X = df[['Sentiment_Score', 'Price_Change']].values
    
    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000)
    model.fit(X)
    
    bullish_state = 0 if model.means_[0][1] > model.means_[1][1] else 1
    
    joblib.dump((model, bullish_state), MODEL_PATH)
    return model, bullish_state

def main():
    if not os.path.exists(DATA_PATH): return
    df = pd.read_csv(DATA_PATH).dropna()
    
    if len(df) < 10:
        print("Waiting for more historical data...")
        return

    model, bullish_state = train_hmm(df)
    
    latest_obs = df[['Sentiment_Score', 'Price_Change']].tail(1).values
    current_state = model.predict(latest_obs)[0]
    
    prob_next_bullish = model.transmat_[current_state][bullish_state]
    
    prediction = "UP" if prob_next_bullish > 0.5 else "DOWN"
    conf = prob_next_bullish if prediction == "UP" else (1 - prob_next_bullish)
    
    with open("predictions_log.txt", "a") as f:
        f.write(f"Date: {df.iloc[-1]['Date']} | Prediction: {prediction} | Confidence: {conf:.2f}\n")

if __name__ == "__main__":
    main()
