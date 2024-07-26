# src/xlstm_preprocessing.py

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------
# Normalise data
# -------------------------------------------------------------------------------------------

def normalise_data_xlstm(data):
  scaler = MinMaxScaler(feature_range=(0, 1))
  return scaler.fit_transform(data.reshape(-1, 1)), scaler

def inverse_normalise_data_xlstm(data, scaler):
  return scaler.inverse_transform(data.cpu().numpy().reshape(-1, 1))

# -------------------------------------------------------------------------------------------
# Create sequences
# -------------------------------------------------------------------------------------------

# Function to create sequences
def create_sequences(data, seq_length, dates):
    xs, ys, date_list = [], [], []

    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        date = dates[i + seq_length]
        xs.append(x)
        ys.append(y)
        date_list.append(date)

    X = np.array(xs)
    y = np.array(ys)
    dates = pd.Series(date_list)

    # Convert to PyTorch tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    return X, y, dates

# -------------------------------------------------------------------------------------------
# Train, Validation and Test split
# -------------------------------------------------------------------------------------------

def _split_data(x, y, dates, set, train_end_date, val_end_date):
    if set == 'train':
        mask = (dates < train_end_date)
    elif set == 'val':
        mask = (dates >= train_end_date) & (dates < val_end_date)
    elif set == 'test':
        mask = (dates >= val_end_date)
    else:
        raise ValueError("Invalid set name. Must be 'train', 'val', or 'test'.")

    # Move data to GPU
    x_splitted = x[mask].to('cuda')
    y_splitted = y[mask].to('cuda')

    print(f"{set} X shape: {x_splitted.shape}")
    print(f"{set} y shape: {y_splitted.shape}")

    return x_splitted, y_splitted, dates[mask]

def _plot_data_split(train_dates, train_y, val_dates, val_y, test_dates, test_y, stock):
    # Plot the data
    plt.figure(figsize=(10, 7))

    plt.plot(train_dates, train_y, label='Train Data', color='blue')
    plt.plot(val_dates, val_y, label='Validation Data', color='green')
    plt.plot(test_dates, test_y, label='Test Data', color='red')

    plt.title(f'{stock} Stock Price - Train, Validation, Test Sets')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

def split_train_val_test_xlstm(x, y, dates, train_end_date, val_end_date, scaler, stock):
    train_x, train_y, train_dates = _split_data(x, y, dates, 'train', train_end_date, val_end_date)
    val_x, val_y, val_dates = _split_data(x, y, dates, 'val', train_end_date, val_end_date)
    test_x, test_y, test_dates = _split_data(x, y, dates, 'test', train_end_date, val_end_date)

    _plot_data_split(train_dates.to_numpy(), inverse_normalise_data_xlstm(train_y, scaler), val_dates.to_numpy(), inverse_normalise_data_xlstm(val_y, scaler), test_dates.to_numpy(), inverse_normalise_data_xlstm(test_y, scaler), stock)

    return train_x, train_y, train_dates, val_x, val_y, val_dates, test_x, test_y, test_dates
