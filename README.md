# Nifty50 Sentiment-Based Prediction Engine

An automated NLP and Machine Learning pipeline that predicts the next-day movement of the **Nifty 50 (NSE)** by analyzing financial news sentiment and historical price action. 

This project was developed for the **Natural Language Processing (NLP)** course to demonstrate the impact of textual sentiment on financial markets.

---

## Key Features
* **NLP Engine:** Uses **FinBERT** (Financial BERT), a pre-trained language model specifically tuned for financial sentiment analysis.
* **Automated Pipeline:** Powered by **GitHub Actions** to fetch news, analyze sentiment, and predict daily at 08:00 AM IST.
* **Self-Correction Logic:** Implements a **Sliding Window Retraining** strategy. If rolling accuracy drops below 55%, the model automatically retrains on the most recent 60 trading days.
* **Data Fusion:** Combines qualitative data (News Headlines) with quantitative data (Moving Averages & Price Change).

---

## Tech Stack
| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.9+ |
| **NLP Model** | [FinBERT](https://huggingface.co/ProsusAI/finbert) (Hugging Face) |
| **Machine Learning** | Random Forest (Scikit-Learn) |
| **Financial Data** | Yahoo Finance API (`yfinance`) |
| **News Source** | NewsAPI.org |
| **Automation** | GitHub Actions |

---

## Project Structure
* `fetch_data.py`: Scrapes financial news and calculates sentiment scores using FinBERT.
* `model.py`: The prediction engine that handles feature engineering and the retraining logic.
* `market_data.csv`: The evolving dataset containing historical prices and sentiment scores.
* `.github/workflows/daily_prediction.yml`: The automation script for daily execution.

---

## How It Works
1.  **Sentiment Extraction:** The system fetches the top 15 headlines related to the Indian Stock Market. FinBERT assigns a weight to each (Positive, Negative, Neutral).
2.  **Feature Engineering:** It calculates the **Daily Sentiment Index** and merges it with the 5-day Moving Average (MA) of Nifty 50.
3.  **Prediction:** A Random Forest Classifier predicts whether the next trading day will close **UP** or **DOWN**.
4.  **Feedback Loop:** After the market closes, the actual result is compared to the prediction. If the model's accuracy trend is declining, it triggers a retrain to adapt to new market "moods."

---

## 🛠️ Setup & Installation
1.  **Clone the Repo:**
    ```bash
    git clone
