import os
import requests
import pandas as pd
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime, timedelta

NEWS_API_KEY = os.getenv('NEWS_API_KEY')
SYMBOL = "^NSEI"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment(headlines):
    """
    Passes headlines through FinBERT and returns a single average score.
    Returns a score between -1 (Very Bearish) and 1 (Very Bullish).
    """
    if not headlines:
        return 0.0
    
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    mean_preds = predictions.mean(dim=0)
    sentiment_score = mean_preds[0].item() - mean_preds[1].item()
    return sentiment_score

def fetch_news():
    """
    Fetches latest Indian stock market news headlines.
    """
    url = f'https://newsapi.org/v2/everything?q=Nifty50+OR+"Indian+Stock+Market"&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}'
    response = requests.get(url).json()
    articles = response.get('articles', [])
    return [a['title'] for a in articles[:15]]

def main():
    nifty = yf.Ticker(SYMBOL)
    hist = nifty.history(period="5d")
    
    today_price = hist['Close'].iloc[-1]
    today_date = hist.index[-1].strftime('%Y-%m-%d')
    
    print("Fetching news and analyzing sentiment...")
    headlines = fetch_news()
    sentiment_score = get_sentiment(headlines)
    
    new_data = {
        'Date': [today_date],
        'Close': [today_price],
        'Sentiment_Score': [sentiment_score]
    }
    new_df = pd.DataFrame(new_data)
    
    if os.path.exists("market_data.csv"):
        old_df = pd.read_csv("market_data.csv")
        updated_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['Date'], keep='last')
        updated_df.to_csv("market_data.csv", index=False)
    else:
        new_df.to_csv("market_data.csv", index=False)
        
    print(f"Data updated for {today_date}. Sentiment Score: {sentiment_score:.2f}")

if __name__ == "__main__":
    main()
