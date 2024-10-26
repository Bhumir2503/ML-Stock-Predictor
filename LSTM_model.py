# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 1: Data Loading and Preprocessing
# Load the stock price data from CSV file
df = pd.read_csv('./data/BRK.csv')

# Convert 'Close' column to numeric format and handle any errors by setting them to NaN
df['Close'] = pd.to_numeric(df.Close, errors='coerce')

# Remove any rows with missing values
df = df.dropna()

# Extract the 'Close' price column for training (column index 4)
trainData = df.iloc[:, 4:5].values

# Step 2: Feature Scaling
# Initialize MinMaxScaler to scale values between 0 and 1
sc = MinMaxScaler(feature_range=(0, 1))

# Transform the training data to scaled version
training_set_scaled = sc.fit_transform(trainData)
trainData.shape  # Display the shape of training data

# Step 3: Data Preparation for LSTM
# Create arrays to store 60 time-steps of historical data and the target value
X_train = []  # Will contain 60 time steps of data
y_train = []  # Will contain the next value to predict

# Sliding window approach to create sequences
for i in range(60, trainData.shape[0]):
    # Append 60 days of prices to X_train
    X_train.append(training_set_scaled[i-60:i, 0])
    # Append the next day's price to y_train
    y_train.append(training_set_scaled[i, 0])

# Convert lists to numpy arrays for training
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape X_train to match LSTM input format: [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Step 4: Building the LSTM Model
# Initialize Sequential model
model = Sequential()

# First LSTM layer
model.add(LSTM(units=50,               # Number of LSTM units/neurons
   return_sequences=True,              # Return full sequence for next LSTM layer
   input_shape=(X_train.shape[1], 1))) # Input shape: (60 timesteps, 1 feature)
model.add(Dropout(0.2))                # Add 20% dropout to prevent overfitting

# Second LSTM layer
model.add(LSTM(units=50, 
   return_sequences=True))             # Return sequences for next LSTM layer
model.add(Dropout(0.2))

# Third LSTM layer
model.add(LSTM(units=50, 
   return_sequences=True))             # Return sequences for next LSTM layer
model.add(Dropout(0.2))

# Fourth LSTM layer
model.add(LSTM(units=50))              # No return_sequences needed for last LSTM layer
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1))              # Single unit for price prediction

# Step 5: Model Compilation
# Configure the model for training
model.compile(optimizer='adam',         # Adam optimizer for adaptive learning rate
   loss='mean_squared_error')           # MSE loss function for regression

# Step 6: Model Training
# Train the model on the prepared data
model.fit(X_train, y_train,            # Training data and labels
   epochs=100,                         # Number of training iterations
   batch_size=32)                      # Number of samples per gradient update

# Step 7: Save the trained model
# Save the model to a file for later use
model.save('./Model/StockPredictorTrained.h5')