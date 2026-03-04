import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

MODEL_PATH = "nifty_model.pkl"
DATA_PATH = "market_data.csv"
RETRAIN_THRESHOLD_ACCURACY = 0.55
WINDOW_SIZE = 60

def prepare_features(df):
    """
    Creates technical indicators to pair with your NLP sentiment.
    """
    df['Price_Change'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()

def train_model(train_data):
    """
    Trains the Random Forest. We use Random Forest because it 
    handles the 'noise' of NLP sentiment better than deep learning.
    """
    X = train_data[['Sentiment_Score', 'Price_Change', 'MA_5']]
    y = train_data['Target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print("Model Retrained Successfully.")
    return model

def main():
    if not os.path.exists(DATA_PATH):
        print("No data found.")
        return

    df = pd.read_csv(DATA_PATH)
    
    if len(df) < 6:
        print(f"Not enough data to train yet (need at least 6 days, currently have {len(df)}).")
        print("Collecting today's data and waiting for more history...")
        return 

    df = prepare_features(df)

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        
        test_df = df.tail(10)
        X_test = test_df[['Sentiment_Score', 'Price_Change', 'MA_5']]
        y_test = test_df['Target']
        
        predictions = model.predict(X_test)
        current_acc = accuracy_score(y_test, predictions)
        
        print(f"Current Rolling Accuracy: {current_acc:.2f}")

        if current_acc < RETRAIN_THRESHOLD_ACCURACY:
            print("Accuracy below threshold. Triggering sliding window retraining...")
            train_data = df.tail(WINDOW_SIZE)
            model = train_model(train_data)
    else:
        print("No model found. Initializing first training...")
        model = train_model(df)

    latest_features = df.tail(1)[['Sentiment_Score', 'Price_Change', 'MA_5']]
    prediction = model.predict(latest_features)[0]
    confidence = model.predict_proba(latest_features)[0]

    result = "UP" if prediction == 1 else "DOWN"
    print(f"PREDICTION FOR NEXT TRADING DAY: {result} ({max(confidence)*100:.2f}% confidence)")

    with open("predictions_log.txt", "a") as f:
        f.write(f"Date: {df.iloc[-1]['Date']} | Prediction: {result} | Confidence: {max(confidence):.2f}\n")

if __name__ == "__main__":
    main()
