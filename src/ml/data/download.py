# src/ml/data/download.py

import datetime
import yfinance as yf
import requests
import pandas as pd
import time
import os
import urllib.parse
from fake_useragent import UserAgent

# Utility function to validate date format (YYYY-MM-DD)
def _check_date_format(date):
    try:
        return datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return None

# Main function to download data from Yahoo Finance or Tiingo
def download_data(ticker, start_date=None, end_date=None, freq='1d'):
    if freq == '1h':
        try:
            # Try to download data from Tiingo API (more than 730 days)
            return _download_tiingo_data(ticker, start_date, end_date)
        except DataDownloadError:
            pass
            # Continue with Yahoo Finance data download (less than 730 days)

    # Fetch data using Yahoo Finance
    data = yf.Ticker(ticker)
    start_date_dt = _check_date_format(start_date)
    end_date_dt = _check_date_format(end_date)
    
    try:
        if freq == '1h':
            if not start_date_dt or (start_date_dt and start_date_dt < end_date_dt - datetime.timedelta(days=730)):
                return data.history(period='2y', interval=freq) # API limitation for intraday data
            return data.history(start=start_date_dt, end=end_date_dt, interval=freq)
        
        if not start_date_dt: # Fetch data from the beginning
            return data.history(end=end_date_dt, period='max', interval=freq)
        else:
            return data.history(start=start_date_dt, end=end_date_dt, interval=freq)
    except (ValueError, TypeError, AttributeError):
        return data.history(period='max', interval=freq)

# Utility function to fetch intraday data from Tiingo API
def _fetch_intraday_data(ticker, start_date, end_date, api_key):
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

# Custom exception to handle data download errors
class DataDownloadError(Exception):
    """Raised when data download fails due to missing API key or API limitation."""
    pass

# Function to handle Tiingo data download with pagination
def _download_tiingo_data(ticker, start_date=None, end_date=None):
    TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')

    # Check if the API key is set
    if not TIINGO_API_KEY:
        raise DataDownloadError("TIINGO_API_KEY is missing. Please set the environment variable.")

    try:
        end_date_dt = end_date.strftime("%Y-%m-%d")
    except (ValueError, TypeError, AttributeError):
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

        try:
            # Fetch the data
            df = _fetch_intraday_data(ticker, chunk_start_date_str, chunk_end_date_str, TIINGO_API_KEY)
            all_data = pd.concat([df, all_data])
        except Exception as e:
            raise DataDownloadError(f"Failed to download data from TIINGO for {ticker}. Error: {str(e)}")

        # Update the end date for the next chunk
        chunk_end_date = chunk_start_date - datetime.timedelta(days=1)

        # Respect API rate limits
        time.sleep(1)

    # Sort the combined data by timestamp in descending order
    all_data.sort_index(ascending=True, inplace=True)

    return all_data

# Function to search for ticker symbols
def search_ticker(search_term):
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
