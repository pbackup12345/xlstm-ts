# src/data_download.py

import datetime
import yfinance as yf
import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
import os

TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')

def download_data(ticker, start_date, end_date):
    data = yf.Ticker(ticker)
    df = data.history(start=start_date, end=end_date)
    return df

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

def download_hourly_data(ticker, years_back):
    # Current date
    end_date = datetime.datetime.now()
    # Start date x years back
    start_date = end_date - datetime.timedelta(days=years_back*365)

    # DataFrame to store the combined data
    all_data = pd.DataFrame()

    # Fetch data in chunks due to API limitations
    chunk_end_date = end_date
    while chunk_end_date > start_date:
        chunk_start_date = chunk_end_date - datetime.timedelta(days=730)
        if chunk_start_date < start_date:
            chunk_start_date = start_date

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

def plot_data(df, stock):
    plt.figure(figsize=(10, 7))
    plt.plot(df.index, df['Close'])
    plt.title(f'{stock} Stock')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.show()
