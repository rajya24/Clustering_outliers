import pandas as pd

def create_time_series_features(df, datetime_col, rolling_window=3, lags=[1, 2, 3]):
    """
    Creates time-based features, rolling statistics, and lag features.

    Parameters:
    - df: pd.DataFrame, the input dataframe containing time series data.
    - datetime_col: str, the name of the datetime column in the dataframe.
    - rolling_window: int, the window size for calculating rolling statistics.
    - lags: list of int, the lag periods to create lag features.

    Returns:
    - df: pd.DataFrame, the dataframe with new time-based, rolling, and lag features.
    """

    # Ensure the datetime column is in datetime format
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Set the datetime column as the index
    df.set_index(datetime_col, inplace=True)

    # Create time-based features
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['week'] = df.index.isocalendar().week
    df['day'] = df.index.day
    df['weekday'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['hour'] = df.index.hour
    df['is_weekend'] = df.index.dayofweek >= 5

    # Create rolling statistics features
    df[f'rolling_mean_{rolling_window}'] = df['value'].rolling(window=rolling_window).mean()
    df[f'rolling_std_{rolling_window}'] = df['value'].rolling(window=rolling_window).std()
    df[f'rolling_min_{rolling_window}'] = df['value'].rolling(window=rolling_window).min()
    df[f'rolling_max_{rolling_window}'] = df['value'].rolling(window=rolling_window).max()

    # Create lag features
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)

    # Reset the index to bring the datetime column back
    df.reset_index(inplace=True)

    return df