# src/ml/pipeline/pipeline.py

from ml.data.download import download_data

def run_pipeline(stock, ticker, start_date, freq="1d"):
    # 1. Data collection
    df = download_data(ticker, start_date, freq=freq)
    print(df.head())

    # 2. Preprocessing

    # 3. Model

    # 4. Training

    # 5. Evaluation

    # 6. Results