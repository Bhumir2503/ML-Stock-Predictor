import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def predict_stock(model_path, stock_data_path, company_name):
    """
    Predict stock prices using a saved model
    
    Parameters:
    model_path (str): Path to the saved model (.h5 file)
    stock_data_path (str): Path to the new stock's CSV data
    company_name (str): Name of the company for plot title
    """
    # Load the saved model
    model = load_model(model_path)
    
    # Load and prepare the new stock data
    df = pd.read_csv(stock_data_path)
    df['Close'] = pd.to_numeric(df.Close, errors='coerce')
    df = df.dropna()
    stock_data = df.iloc[:, 4:5].values  # Get Close prices
    
    # Feature Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)
    
    # Prepare data for prediction (last 60 days)
    X_test = []
    for i in range(60, len(scaled_data)):
        X_test.append(scaled_data[i-60:i, 0])
    X_test = np.array(X_test)
    
    # Reshape for LSTM input
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Make predictions
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    # Prepare actual prices for comparison
    real_prices = stock_data[60:]
    
    # Calculate prediction metrics
    mse = np.mean((real_prices - predicted_prices) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(real_prices - predicted_prices))
    
    # Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(real_prices, color='red', label=f'Real {company_name} Stock Price')
    plt.plot(predicted_prices, color='blue', label=f'Predicted {company_name} Stock Price')
    plt.title(f'{company_name} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    
    # Add prediction metrics to plot
    plt.figtext(0.15, 0.15, f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}', 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()
    
    # Predict next 5 days
    last_60_days = scaled_data[-60:]
    future_predictions = []
    
    current_batch = last_60_days.reshape((1, 60, 1))
    for _ in range(5):
        # Get prediction for next day
        next_pred = model.predict(current_batch)[0]
        future_predictions.append(next_pred)
        
        # Update batch for next prediction
        current_batch = np.roll(current_batch, -1, axis=1)
        current_batch[0, -1, 0] = next_pred

    # Transform predictions back to original scale
    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    # Get dates for future predictions
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    future_dates = pd.date_range(start=last_date, periods=6)[1:]
    
    print("\nPredicted prices for next 5 trading days:")
    for date, price in zip(future_dates, future_prices):
        print(f"{date.date()}: ${price[0]:.2f}")
        
    return real_prices, predicted_prices, future_prices


if __name__ == "__main__":
    # Parameters
    MODEL_PATH = './Model/MainPredictor.h5'
    NEW_STOCK_DATA = './data/BAC.csv'
    COMPANY_NAME = 'BAC'
    
    # Make predictions
    real_prices, predicted_prices, future_prices = predict_stock(
        MODEL_PATH, 
        NEW_STOCK_DATA, 
        COMPANY_NAME
    )