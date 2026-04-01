import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import joblib
import os

MODEL_PATH = "nifty_hmm.pkl"
DATA_PATH = "market_data.csv"

def train_model(df):
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
        print("Waiting for more data history...")
        return

    model, bullish_state = train_model(df)
    
    current_obs = df[['Sentiment_Score', 'Price_Change']].tail(1).values
    current_state = model.predict(current_obs)[0]
    prob_up = model.transmat_[current_state][bullish_state]
    
    prediction = "UP" if prob_up > 0.5 else "DOWN"
    confidence = prob_up if prediction == "UP" else (1 - prob_up)
    
    output_line = f"Date: {df.iloc[-1]['Date']} | Prediction: {prediction} | Confidence: {confidence:.2f}\n"
    
    with open("predictions_log.txt", "a") as f:
        f.write(output_line)
    print(output_line)

if __name__ == "__main__":
    main()
