import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import mlflow
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import pickle
import os

# Create dummy data for EBIT, Interest Income, and Interest Expense
dates = pd.date_range(start='2017-01-01', periods=28, freq='Q')
np.random.seed(42)
data = {
    'Date': dates,
    'EBIT': np.random.normal(loc=100, scale=10, size=len(dates)),
    'Interest_Income': np.random.normal(loc=10, scale=2, size=len(dates)),
    'Interest_Expense': np.random.normal(loc=5, scale=1, size=len(dates))
}

df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Split data into train and test
train_size = len(df) - 4
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to train and forecast using ARIMA model
def train_and_forecast_arima(train_data, test_data, order=(1, 1, 1)):
    model = sm.tsa.ARIMA(train_data, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test_data))
    return model_fit, forecast

# Function to calculate accuracy for time series
def calculate_accuracy(test, forecast):
    return np.mean(np.abs((test - forecast) / test)) * 100

# Function to plot and save ACF and PACF
def save_acf_pacf_plots(data, column_name, nlags):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    plot_acf(data, ax=axes[0], lags=nlags)
    axes[0].set_title(f'{column_name} ACF')
    
    plot_pacf(data, ax=axes[1], lags=nlags)
    axes[1].set_title(f'{column_name} PACF')
    
    file_path = f"{column_name}_acf_pacf.png"
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    
    return file_path

# Function to plot and save seasonal decomposition
def save_seasonal_decompose_plot(data, column_name):
    decomposition = seasonal_decompose(data, model='additive', period=4)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    
    file_path = f"{column_name}_seasonal_decompose.png"
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    
    return file_path

# Initialize MLflow
mlflow.set_experiment("India")

# Create a directory to store the pickle files
os.makedirs("models", exist_ok=True)

# Train and evaluate models using ARIMA from statsmodels
results = {}

for column in df.columns:
    with mlflow.start_run(run_name=column):
        # Train ARIMA model
        model_fit, forecast = train_and_forecast_arima(train[column], test[column])
        
        # Ensure forecast is a NumPy array
        forecast = np.array(forecast)
        
        # Calculate metrics
        train_rmse = rmse(train[column], model_fit.fittedvalues)
        test_rmse = rmse(test[column], forecast)
        accuracy = calculate_accuracy(test[column], forecast)
        
        # Log metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("accuracy", accuracy)

        # Log model parameters
        mlflow.log_param("order", model_fit.model.order)
        mlflow.log_param("aic", model_fit.aic)
        mlflow.log_param("bic", model_fit.bic)
        
        # Save and log ACF and PACF plots
        nlags = min(len(train[column]) // 2 - 1, 10)  # Ensure nlags is valid
        acf_pacf_plot_path = save_acf_pacf_plots(train[column], column, nlags)
        mlflow.log_artifact(acf_pacf_plot_path)
        
        # Save and log seasonal decomposition plot
        seasonal_plot_path = save_seasonal_decompose_plot(train[column], column)
        mlflow.log_artifact(seasonal_plot_path)
        
        # Save the model as a pickle file
        model_path = f"models/{column}_arima_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_fit, f)
        
        # Log the pickle file as an artifact
        mlflow.log_artifact(model_path)
        
        results[column] = {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "accuracy": accuracy,
            "forecast": forecast,
            "order": model_fit.model.order,
            "aic": model_fit.aic,
            "bic": model_fit.bic
        }

print(results)


