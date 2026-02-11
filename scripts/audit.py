import yfinance as yf
import pandas as pd
import sys

def perform_audit():
    try:
        log_df = pd.read_csv('data/prediction_log.csv')
        last_prediction = log_df.iloc[-1]
    except FileNotFoundError:
        print("No prediction log found. Skipping audit.")
        return

    nifty = yf.Ticker("^NSEI")
    hist = nifty.history(period="5d")
    
    actual_close = hist['Close'].iloc[-2] 
    actual_open = hist['Open'].iloc[-2]
    actual_direction = 1 if actual_close > actual_open else 0
    
    predicted_direction = last_prediction['predicted_label']
    
    error_magnitude = abs(actual_close - last_prediction['predicted_price']) / actual_close
    
    print(f"Audit: Actual {actual_close}, Predicted {last_prediction['predicted_price']}")
    
    if actual_direction != predicted_direction or error_magnitude > 0.01:
        print("Threshold breached! Model requires retraining.")
        sys.exit(1)
    else:
        print("Model performed within acceptable limits. No retraining needed.")
        sys.exit(0)

if __name__ == "__main__":
    perform_audit()
