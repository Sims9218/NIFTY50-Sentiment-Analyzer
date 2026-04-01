import os
import requests
import pandas as pd
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

NEWS_API_KEY = os.getenv('NEWS_API_KEY')
SYMBOL = "^NSEI" 

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment(headlines):
    if not headlines: return 0.0
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions.mean(dim=0)[0].item() - predictions.mean(dim=0)[1].item()

def main():
    nifty = yf.Ticker(SYMBOL)
    hist = nifty.history(period="10d")
    
    new_data = {
        'Date': [hist.index[-1].strftime('%Y-%m-%d')],
        'Close': [hist['Close'].iloc[-1]],
        'Price_Change': [hist['Close'].pct_change().iloc[-1]],
        'Sentiment_Score': [get_sentiment([a['title'] for a in requests.get(f'https://newsapi.org/v2/everything?q=Nifty50&apiKey={NEWS_API_KEY}').json().get('articles', [])[:15]])]
    }
    
    df = pd.DataFrame(new_data)
    if os.path.exists("market_data.csv"):
        pd.concat([pd.read_csv("market_data.csv"), df]).drop_duplicates(subset=['Date']).to_csv("market_data.csv", index=False)
    else:
        df.to_csv("market_data.csv", index=False)

if __name__ == "__main__":
    main()
