# src/ml/data/download.py

import datetime
import yfinance as yf
import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import urllib.parse
from fake_useragent import UserAgent

def _search_yahoo_finance(search_term):
    encoded_search_term = urllib.parse.quote(search_term)

    # Construct the URL with the search term
    search_url = f'https://query2.finance.yahoo.com/v1/finance/search?q={encoded_search_term}&lang=en-US&region=US&quotesCount=8&quotesQueryId=tss_match_phrase_query&multiQuoteQueryId=multi_quote_single_token_query&enableCb=true&enableNavLinks=true&enableCulturalAssets=true&enableNews=false&enableResearchReports=false&researchReportsCount=2&newsCount=0'
    
    # Get the most updated User-Agent for Chrome
    ua = UserAgent()
    HEADERS = {
        'User-Agent': ua.chrome
    }

    # Execute the GET request
    response = requests.get(search_url, headers=HEADERS)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        results = response.json()
        
        search_results = []
        for quote in results.get('quotes', []):
            symbol = quote.get('symbol')
            name = quote.get('shortname')

            if symbol and name:
                search_results.append({'symbol': symbol, 'name': name})
        
        return search_results
    else:
        print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")
        return []
    
def _search_tiingo(search_term):
    TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')

    encoded_search_term = urllib.parse.quote(search_term)

    search_url = f"https://api.tiingo.com/tiingo/utilities/search?query={encoded_search_term}&token={TIINGO_API_KEY}"

    response = requests.get(search_url)
    
    if response.status_code == 200:
        results = response.json()
        search_results = [{'symbol': item['ticker'], 'name': item['name']} for item in results]
        return search_results
    else:
        print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")
        return []
    
def search_ticker(search_term, freq='daily'):

    return _search_yahoo_finance(search_term)

    # The search with Tiingo remains deactivated due to the need for an API key
    #if freq == 'daily':
    #    return _search_yahoo_finance(search_term)
    #elif freq == 'hourly':
    #    return _search_tiingo(search_term)
    #else:
    #    raise ValueError("Invalid frequency. Must be 'daily' or 'hourly'.")

def _download_yahoo_finance_data(ticker, start_date=None, interval='1d'):
    data = yf.Ticker(ticker)

    try:
        start_date_dt = start_date.strftime("%Y-%m-%d")
    except (ValueError, TypeError, AttributeError):
        return data.history(period='max', interval=interval)
    
    end_date_dt = datetime.datetime.now()
    return data.history(start=start_date_dt, end=end_date_dt, interval=interval)

def _download_daily_data(ticker, start_date=None):
    data = yf.Ticker(ticker)

    try:
        start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return data.history(period='2y')
    
    end_date_dt = datetime.datetime.now()
    return data.history(start=start_date_dt, end=end_date_dt)

def fetch_intraday_data(ticker, start_date, end_date, api_key):
    url = f'https://api.tiingo.com/iex/{ticker}/prices?startDate={start_date}&endDate={end_date}&resampleFreq=1hour&token={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.index.name = 'Date'
            df.columns = [col.capitalize() for col in df.columns]
        return df
    else:
        print(f"Failed to fetch data for {start_date} to {end_date}")
        return pd.DataFrame()

def _download_hourly_data(ticker, start_date=None):
    TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')

    end_date_dt = datetime.datetime.now()
    try:
        start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        start_date_dt = end_date_dt - datetime.timedelta(days=730)

    # DataFrame to store the combined data
    all_data = pd.DataFrame()

    # Fetch data in chunks due to API limitations
    chunk_end_date = end_date_dt
    while chunk_end_date > start_date_dt:
        chunk_start_date = chunk_end_date - datetime.timedelta(days=730)
        if chunk_start_date < start_date_dt:
            chunk_start_date = start_date_dt

        # Convert dates to string format required by API
        chunk_start_date_str = chunk_start_date.strftime('%Y-%m-%d')
        chunk_end_date_str = chunk_end_date.strftime('%Y-%m-%d')

        # Fetch the data
        df = fetch_intraday_data(ticker, chunk_start_date_str, chunk_end_date_str, TIINGO_API_KEY)
        all_data = pd.concat([df, all_data])

        # Update the end date for the next chunk
        chunk_end_date = chunk_start_date - datetime.timedelta(days=1)

        # Respect API rate limits
        time.sleep(1)

    # Sort the combined data by timestamp in descending order
    all_data.sort_index(ascending=True, inplace=True)

    return all_data

def download_data_gui(ticker, start_date, freq="daily"):
    if freq == "daily":
        df = _download_yahoo_finance_data(ticker, start_date, interval='1d')
    elif freq == "hourly":
        df = _download_yahoo_finance_data(ticker, start_date, interval='1h')
    else:
        raise ValueError("Invalid frequency. Must be 'daily' or 'hourly'.")

    return df

def download_data(ticker, start_date, freq="daily"):
    if freq == "daily":
        df = _download_daily_data(ticker, start_date)
    elif freq == "hourly":
        df = _download_hourly_data(ticker, start_date)
    else:
        raise ValueError("Invalid frequency. Must be 'daily' or 'hourly'.")

    return df

def plot_data(df, stock, ax=None):
    if ax:
        ax.clear()
        ax.plot(df.index, df['Close'], label=stock)
        ax.set_title(f'{stock} Price Data')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
    else:
        plt.figure(figsize=(10, 7))
        plt.plot(df.index, df['Close'], label=stock)
        plt.title(f'{stock} Price Data')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.show()
