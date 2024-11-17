import pandas as pd
import pandas_ta as ta
import numpy as np

def create_features(df, classifier=False):
    '''
    Takes raw data from dataframe and returns new dataFrame with the desired features and target.
        
    This version uses lagged closing price values, 10MA, 50MA, 200MA, RSI, and the 3 relevant MACD indicators.
    '''

    features = pd.DataFrame({})

    # Change target feature based on prediction type
    if classifier:
        # Closing price difference between days
        # Also have to remove first value from Close/Date because diff removes a value
        # Positive = 1 , Negative (or 0) = 0
        features['Close'] = df['Close'].tail(-1)
        features['Date'] = df['Date'].tail(-1)
        features['Up/Down'] = (np.diff(df['Close']) > 0).astype(int)
    else:
        features['Close'] = df['Close']
        features['Date'] = df['Date']

    # Create lagged price features (Use closing price from days 0-6 to predict closing price of day 7)
    for day in range(1,8):
        features[f'Close Day -{day}'] = df['Close'].shift(day)

    # The rolling averages below need to be shifted because they include the current day
    # in the average calculation, which would create leakage.
    ## SHORT TERM TREND ##

    # Create 10 day moving average feature
    features['10MA'] = features['Close'].rolling(10).mean()

    # Lag the feature so that the average of days 0-9 is used to predict day 10
    features['10MA'] = features['10MA'].shift(1)

    ## MEDIUM TERM TREND ##

    # Create 50 day moving average feature
    features['50MA'] = features['Close'].rolling(50).mean()

    # Lag the feature so that the average of days 0-49 is used to predict day 50
    features['50MA'] = features['50MA'].shift(1)

    ## LONG TERM TREND ##

    # Create 200 day moving average feature
    features['200MA'] = features['Close'].rolling(200).mean()

    # Lag the feature so that the average of days 0-199 is used to predict day 200
    features['200MA'] = features['200MA'].shift(1)

    # Create relative strength indicator feature (defaults to 14 day)
    # This one doesn't need to be lagged because it is inherently calculated based on the past 14 days
    features['RSI'] = ta.rsi(features['Close'])

    # MACD uses exponential moving averages (EMAs), there is some advantage to using EMAs over MAs but idk
    # MACD is the difference between a 12 day and 26 day EMA, such that a positive MACD implies upward trend and vice versa

    # There are 3 parts to it, the MACD line (which is the 12EMA-26EMA), the signal line (which is a 9 day EMA of the MACD line itself)
    # and the MACD histogram, which is the difference between the MACD and the signal line.

    # When the MACD line crosses above the signal line, it indicates a buy signal and vice versa as the momentum is shifting
    # The histogram indicates the strength of the momentum

    macd = ta.macd(features['Close']).shift(1)

    features['MACD'] = macd['MACD_12_26_9']
    features['MACD Hist'] = macd['MACDh_12_26_9']
    features['MACD Signal'] = macd['MACDs_12_26_9']

    # Remove NaN values created by 200MA
    features = features.truncate(200)

    if classifier: 
        features = features.drop('Close', axis=1)

    return features