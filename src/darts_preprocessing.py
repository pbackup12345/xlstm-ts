# src/darts_preprocessing.py

import datetime
import pandas as pd
from darts import TimeSeries
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from darts.dataprocessing.transformers.scaler import Scaler

# -------------------------------------------------------------------------------------------
# Convert to darts TimeSeries
# -------------------------------------------------------------------------------------------

def _get_ts_series(times, values):
    series = TimeSeries.from_times_and_values(times=times, values=values)
    assert not pd.isna(series.pd_dataframe()).any().any(), "Darts time series contains NaN values"
    return series

def _adjust_for_dst(df, freq, min_hour):
    df = df.copy()
    df['date'] = df.index.normalize()
    adjusted_dates = []

    for _, group in df.groupby('date'):
        hour_diff = group.index[0].hour - min_hour.hour

        if hour_diff == 1:  # Transition to DST (Spring forward)
            adjusted_group = group.index - datetime.timedelta(hours=1)
            adjusted_dates.extend(adjusted_group)
        elif hour_diff == -1:  # Transition back to standard time (Fall back)
            adjusted_group = group.index + datetime.timedelta(hours=1)
            adjusted_dates.extend(adjusted_group)
        else:  # No change
            adjusted_dates.extend(group.index)
    
    return pd.DatetimeIndex(adjusted_dates, freq=freq)

def convert_to_ts_hourly(df):
    # Find the default business hours
    min_date = df.index.min().normalize()
    df_min_date = df[df.index.normalize() == min_date]

    min_hour = df_min_date.index.min()
    max_hour = (df_min_date.index.max() + datetime.timedelta(hours=1))

    min_hour_str = str(min_hour.strftime('%H:%M'))
    max_hour_str = str(max_hour.strftime('%H:%M'))

    # Create a complete date range including all business days within the range
    min_date = df.index.min()
    max_date = df.index.max()
    
    bhour_mon = pd.offsets.CustomBusinessHour(start=min_hour_str, end=max_hour_str)
    complete_date_range = pd.date_range(start=min_date, end=max_date, freq=bhour_mon)

    # Adjust the index
    adjusted_index = _adjust_for_dst(df, bhour_mon, min_hour)

    # Apply the adjusted index to the DataFrame
    df.index = adjusted_index

    # Sort the DataFrame by the new index to ensure order
    df = df.sort_index()

    series = _get_ts_series(times=complete_date_range, values=df['Close'].values)
    series_denoised = _get_ts_series(times=complete_date_range, values=df['Close_denoised'].values)

    return series, series_denoised

def convert_to_ts_daily(df):
    # Create a complete date range including all business days within the range
    min_date = df.index.min()
    max_date = df.index.max()
    complete_date_range = pd.date_range(start=min_date, end=max_date, freq='B')

    # Identify missing dates
    existing_dates = df.index
    missing_dates = complete_date_range.difference(existing_dates)

    my_holidays = missing_dates.strftime('%Y-%m-%d').tolist()

    # Convert the array of dates to datetime format
    my_holidays = pd.to_datetime(my_holidays)

    # Create a list of Holiday objects
    holiday_rules = [Holiday(name=f"CustomHoliday{idx}", year=date.year, month=date.month, day=date.day, observance=nearest_workday) for idx, date in enumerate(my_holidays)]

    # Create a custom holiday calendar
    class MyHolidayCalendar(AbstractHolidayCalendar):
        rules = holiday_rules

    # Create the custom business day offset
    custom_bday = pd.offsets.CustomBusinessDay(calendar=MyHolidayCalendar())

    # To use this custom business day offset
    times = pd.DatetimeIndex(df.index, freq=custom_bday)

    series = _get_ts_series(times=times, values=df['Close'])
    series_denoised = _get_ts_series(times=times, values=df['Close_denoised'])

    return series, series_denoised

# -------------------------------------------------------------------------------------------
# Train, Validation and Test split
# -------------------------------------------------------------------------------------------

def split_train_val_test_darts(series, train_end_date, val_end_date):
    # Split the data by date
    train, temp = series.split_before(pd.Timestamp(train_end_date))

    val, test = temp.split_before(pd.Timestamp(val_end_date))

    return train, val, test

# -------------------------------------------------------------------------------------------
# Normalise data
# -------------------------------------------------------------------------------------------

def normalise_data_darts(data, scaler=None):
    if scaler:
        return scaler.transform(data)
    else:
        scaler = Scaler() # default uses sklearn's MinMaxScaler
        data = scaler.fit_transform(data)
        return data, scaler
  
def inverse_normalise_data_darts(data, scaler):
    return scaler.inverse_transform(data)

def normalise_split_data_darts(train, val, test):
    train, scaler = normalise_data_darts(train)
    val = normalise_data_darts(val, scaler)
    test = normalise_data_darts(test, scaler)

    return train, val, test, scaler
