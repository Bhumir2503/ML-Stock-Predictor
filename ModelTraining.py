import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from datetime import datetime
import os

class StockModelTrainer:
   def __init__(self, model_path='apple_stock_prediction.h5', learning_rate=0.001):
      """
      Initialize the trainer with an existing model
      Args:
         model_path: Path to the existing trained model
         learning_rate: Learning rate for the optimizer
      """
      # Load the model
      self.model = load_model(model_path, compile=False)  # Load without compilation
        
      # Recompile the model with a fresh optimizer
      self.model.compile(
         optimizer=Adam(learning_rate=learning_rate),
         loss='mean_squared_error'
      )
        
      self.scaler = MinMaxScaler(feature_range=(0, 1))
      self.sequence_length = 60  # Number of time steps the model expects
        
   def prepare_data(self, data_path, price_column='Close'):
      """
      Prepare new data for training
      Args:
         data_path: Path to CSV file containing new data
         price_column: Name of the column containing price data
      Returns:
         Prepared X and y data for training
      """
      print(f"Loading data from: {data_path}")
        
      # Load and preprocess the new data
      df = pd.read_csv(data_path)
        
      # Ensure the price column exists
      if price_column not in df.columns:
         raise ValueError(f"Column '{price_column}' not found in the dataset. Available columns: {df.columns.tolist()}")
            
      # Convert price column to numeric and handle missing values
      df[price_column] = pd.to_numeric(df[price_column], errors='coerce')
      initial_rows = len(df)
      df = df.dropna(subset=[price_column])
      dropped_rows = initial_rows - len(df)
      if dropped_rows > 0:
         print(f"Dropped {dropped_rows} rows with missing values")
            
      # Extract and reshape the price data
      price_data = df[price_column].values.reshape(-1, 1)
        
      # Fit or transform the data
      if not hasattr(self.scaler, 'n_features_in_'):
         print("Fitting scaler to new data...")
         scaled_data = self.scaler.fit_transform(price_data)
      else:
         print("Transforming data with existing scaler...")
         scaled_data = self.scaler.transform(price_data)
        
      # Prepare sequences
      X, y = [], []
      for i in range(self.sequence_length, len(scaled_data)):
         X.append(scaled_data[i-self.sequence_length:i, 0])
         y.append(scaled_data[i, 0])
        
      X, y = np.array(X), np.array(y)
      print(f"Prepared {len(X)} sequences for training")
        
      return X, y
    
   def continue_training(self, X, y, epochs=100, batch_size=32, validation_split=0.1):
      """
      Continue training the model with new data
      Args:
         X: Input sequences
         y: Target values
         epochs: Number of training epochs
         batch_size: Size of training batches
         validation_split: Proportion of data to use for validation
      """
      # Reshape X to match the expected input shape [samples, time steps, features]
      X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
      # Add early stopping to prevent overfitting
      early_stopping = EarlyStopping(
         monitor='val_loss',
         patience=10,
         restore_best_weights=True
      )
        
      # Continue training
      try:
         history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
         )
         return history
      except Exception as e:
         print(f"Error during training: {str(e)}")
         raise
    
   def save_model(self, custom_name=None):
      """
      Save the updated model
      Args:
         custom_name: Custom name for the saved model
      """
      if custom_name is None:
         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
         custom_name = f"./Model/MainPredictor.h5"
            
      self.model.save(custom_name)
      print(f"Model saved as: {custom_name}")
        
   def predict_sequence(self, input_sequence):
      """
      Make predictions using the model
      Args:
         input_sequence: Sequence of prices to base prediction on
      Returns:
         Predicted next price
      """
      # Ensure input sequence is correct length and scaled
      if len(input_sequence) != self.sequence_length:
         raise ValueError(f"Input sequence must be {self.sequence_length} timesteps")
            
      # Scale the input sequence
      scaled_sequence = self.scaler.transform(np.array(input_sequence).reshape(-1, 1))
        
      # Reshape for prediction
      X = np.reshape(scaled_sequence, (1, self.sequence_length, 1))
        
      # Make prediction and inverse scale
      scaled_prediction = self.model.predict(X)
      prediction = self.scaler.inverse_transform(scaled_prediction)
        
      return prediction[0][0]

def train_with_new_data(model_path, data_paths, epochs_per_dataset=100):
   """
   Train the model with multiple new datasets
   Args:
      model_path: Path to the existing model
      data_paths: List of paths to CSV files containing new data
      epochs_per_dataset: Number of epochs to train on each dataset
   """
   # Initialize trainer with fresh optimizer
   trainer = StockModelTrainer(model_path, learning_rate=0.001)
    
   for path in data_paths:
      print(f"\nProcessing dataset: {path}")
      try:
         # Prepare the data
         X, y = trainer.prepare_data(path)
            
         # Continue training
         history = trainer.continue_training(
            X, y,
            epochs=epochs_per_dataset,
            batch_size=32,
            validation_split=0.1
         )
            
         print(f"Training completed for {path}")
            
      except Exception as e:
         print(f"Error processing {path}: {str(e)}")
         continue
    
   # Save the final model
   trainer.save_model()

# Example usage
if __name__ == "__main__":
   model_path = './Model/MainPredictor.h5'
   data_paths = ['./data/AMZN.csv']
   train_with_new_data(model_path, data_paths, epochs_per_dataset=100)


# Already Trained
# BRK
# GOOG
# AAPL
# KO
# MSFT
# BAC
# TSLA
# AMZN