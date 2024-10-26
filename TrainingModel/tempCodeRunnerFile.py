import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Load the CSV file
def load_data(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, parse_dates=['Date'])

    # Sort by date to maintain the time sequence
    df = df.sort_values('Date')
    
    # Selecting relevant columns (Features and Target)
    X = df[['Open', 'High', 'Low', 'Volume']]  # Features
    y = df['Close']  # Target (Close price)

    return X, y, df['Date']

# Prepare the data for training
def prepare_data(X, y):
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scale the features (important for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

# Train the KNN model
def train_knn(X_train, y_train, n_neighbors=5):
    # Create the KNN Regressor model
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)

    # Fit the model to the training data
    knn.fit(X_train, y_train)

    return knn

# Evaluate the model on the test data
def evaluate_model(knn, X_test, y_test):
    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Calculate Mean Squared Error (MSE) to evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.4f}')

    return y_pred

# Plot the actual vs predicted stock prices
def plot_predictions(dates, y_test, y_pred):
    plt.figure(figsize=(10,6))

    # Plot actual Close prices
    plt.plot(dates, y_test, label='Actual Close Prices', color='blue')

    # Plot predicted Close prices
    plt.plot(dates, y_pred, label='Predicted Close Prices', color='red')

    plt.title('Actual vs Predicted Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main function
def main(file_path):
    # Load the data
    X, y, dates = load_data(file_path)

    # Prepare the data
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Train the KNN model
    knn = train_knn(X_train, y_train)

    # Evaluate the model
    y_pred = evaluate_model(knn, X_test, y_test)

    # Plot predictions vs actual values
    plot_predictions(dates[-len(y_test):], y_test, y_pred)  # Plot only the test set portion



file_path = './data/BRK.csv'
main(file_path)
