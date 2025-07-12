import os
from datetime import datetime, timedelta
import pandas as pd
import requests
from dotenv import load_dotenv

def get_stock_data(symbol, years=5):
    load_dotenv()
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        raise ValueError("API key not found. Please set ALPHA_VANTAGE_API_KEY in your .env file")
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': 'full',
        'apikey': api_key,
        'datatype': 'json'
    }
    print(f"Fetching data for {symbol}...")
    response = requests.get(url, params=params)
    data = response.json()
    if 'Error Message' in data:
        raise Exception(f"API Error: {data['Error Message']}")
    if 'Time Series (Daily)' not in data:
        raise Exception("Unexpected API response format")
    df = pd.DataFrame(data['Time Series (Daily)']).T
    df.index = pd.to_datetime(df.index)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    df = df.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume'
    })
    df = df.sort_index()
    if years:
        cutoff_date = datetime.now() - timedelta(days=years*365)
        df = df[df.index >= cutoff_date]
    return df

def save_to_csv(df, filename=None):
    if filename is None:
        filename = f"stock_data_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(filename)
    print(f"Data saved to {filename}")
    return filename

if __name__ == "__main__":
    try:
        symbol = "NXPI"
        years = 5
        df = get_stock_data(symbol, years)
        print(f"\nFetched {len(df)} days of data for {symbol}")
        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
        print("\nFirst 5 rows:")
        print(df.head())
        filename = f"{symbol}_{years}year_data.csv"
        save_to_csv(df, filename)        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
