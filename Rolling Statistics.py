import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def treat_time_series_outliers(time_series, window_size=5, sigma=2):
    """
    Identifies and treats outliers in a time series using rolling mean and standard deviation.

    Parameters:
    time_series: pd.Series, the time series data
    window_size: int, the window size for calculating rolling statistics
    sigma: int, the number of standard deviations for defining outliers

    Returns:
    time_series_cleaned: pd.Series, the time series after outlier treatment
    outliers: pd.Series, the identified outliers
    """

    # Calculate rolling statistics
    rolling_mean = time_series.rolling(window=window_size).mean()
    rolling_std = time_series.rolling(window=window_size).std()

    # Identify outliers
    outliers = time_series[(time_series < rolling_mean - sigma * rolling_std) |
                           (time_series > rolling_mean + sigma * rolling_std)]

    # Treat outliers by replacing them with rolling mean
    time_series_cleaned = time_series.copy()
    time_series_cleaned[outliers.index] = rolling_mean[outliers.index]

    # Plotting the original and cleaned data
    plt.figure(figsize=(14,7))
    plt.plot(time_series, color='blue', label='Original Data')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.scatter(outliers.index, outliers, color='orange', label='Outliers', marker='o')
    plt.title('Time Series Data with Rolling Mean and Outliers')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14,7))
    plt.plot(time_series_cleaned, color='green', label='Cleaned Data')
    plt.title('Time Series Data after Outlier Treatment')
    plt.legend()
    plt.show()

    return time_series_cleaned, outliers