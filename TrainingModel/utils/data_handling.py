import pandas as pd

def load_data(file_path):
    '''
    Simple data load with whitespace stripping, NaN dropping, and datetime conversion
    '''

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])

    return df

def sequential_split(X, y, dates, test_size=None):
    '''
    Split data sequentially to preserve order.
    Works similarly to train_test_split but
    without randomization.

    Defaults to test size of 1 day if not provided.
    '''
    if test_size==None:
        test_size = 1/len(X)

    splitIndex = int(len(X)*(1-test_size))
    
    X_train = X[:splitIndex]
    X_test = X[splitIndex:]

    y_train = y[:splitIndex]
    y_test = y[splitIndex:]

    dates_train = dates[:splitIndex]
    dates_test = dates[splitIndex:]

    df = {
        'X_train' : X_train,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test,
        'dates_train' : dates_train,
        'dates_test' : dates_test
        }
    
    print(f'\nTraining on {min(dates_train).date()} to {max(dates_train).date()}')
    print(f'\nTesting on {min(dates_test).date()} to {max(dates_test).date()}\n')

    return df