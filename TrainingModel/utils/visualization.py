import matplotlib.pyplot as plt
import pandas as pd


def basic_stock_plot(stock_df, stock_ticker):

    '''Simple plot function to visualize a stock's price over time'''

    plt.figure(figsize=(12,8))
    plt.plot(stock_df['Date'], stock_df['High'], color = 'red', alpha=0.5)
    title_string = 'Visualization of ' + stock_ticker + ' Stock Price'
    plt.title(title_string)
    plt.xlabel('Date')
    plt.ylabel('Share Price')
    plt.show()


def comp_table(actual_prices, predicted_prices, stock_ticker):

    '''Format and print a table to compare actual vs. predicted prices'''

    comparison = pd.DataFrame({'Actual': actual_prices.flatten(), 'Predicted': predicted_prices.flatten()})
    table_title = stock_ticker + ' Prediction Comparison'
    print(table_title)
    print(comparison)


def plot_split(split_data):

    '''Basic data split visualization'''

    dates_train = split_data['dates_train']
    dates_test = split_data['dates_test']
    y_train = split_data['y_train']
    y_test = split_data['y_test']

    plt.figure(figsize=(12,8))
    plt.plot(dates_train, y_train, color='blue', label='Training Set')
    plt.plot(dates_test, y_test, color='red', label='Test Set')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.show()

###Future function ideas###

#def plot_metrics (finding best R2, MSE, etc for different hyperparameters, test sizes, etc.)
