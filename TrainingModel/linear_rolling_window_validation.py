import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from utils.data_handling import sequential_split, load_data
from utils.feature_creation import create_features
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
    

path = "../data/BRK.csv"

df = load_data(path)

features = create_features(df, classifier=False)
features=features.dropna()

X = features.drop(columns=['Close', 'Date'])
y = features['Close']

lin_model = LinearRegression()
rfmodel = RandomForestRegressor(n_estimators=1000, max_depth=10, random_state=42)

fold_length = np.floor(len(X)/10).astype(int)   #Prints 220 (roughly 10% of dataset)
training_length = (np.floor(len(X)/10)*5).astype(int)  #Prints 1100 (roughly 50% of dataset)

lin_acc_list = []   #Empty list to store directional accuracies of linear model
rf_acc_list = []    #Empty list to store directional accuracies of random forest model

'''Perform a rolling window cross validation where initial 50% of data is used for training and 
   the next 10% is used for testing. On iteration 2, data range from 10%-60% is used for training
   and range from 60% - 70% is used for testing, and so on.. '''
i = 0
for j in range(1, 6):
    #print(X[(i*fold_length):training_length+(i*fold_length)])
    #print(y[(i*fold_length):training_length+(i*fold_length)])
    training_subset_X = X[(i*fold_length):training_length+(i*fold_length)]
    training_subset_y = y[(i*fold_length):training_length+(i*fold_length)]
    #print(X[training_length+(fold_length*i):training_length+(fold_length*j)])
    #print(y[training_length+(fold_length*i):training_length+(fold_length*j)])
    testing_subset_X = X[training_length+(fold_length*i):training_length+(fold_length*j)]
    testing_subset_y = y[training_length+(fold_length*i):training_length+(fold_length*j)]

    '''Fitting and testing 5 different data subsets for the linear model'''
    lin_model.fit(training_subset_X, training_subset_y)
    validation_set_predictions = lin_model.predict(testing_subset_X)
    predicted_direction = (np.diff(validation_set_predictions) > 0).astype(int)
    actual_direction = (np.diff(testing_subset_y) > 0).astype(int)
    dir_acc = accuracy_score(actual_direction, predicted_direction) * 100
    #print(dir_acc)
    lin_acc_list.append(dir_acc)

    '''Fitting and testing 5 different data subsets for the random forest model'''
    rfmodel.fit(training_subset_X, training_subset_y)
    validation_set_predictions = rfmodel.predict(testing_subset_X)
    predicted_direction = (np.diff(validation_set_predictions) > 0).astype(int)
    actual_direction = (np.diff(testing_subset_y) > 0).astype(int)
    dir_acc = accuracy_score(actual_direction, predicted_direction) * 100
    #print(dir_acc)
    rf_acc_list.append(dir_acc)
    i+=1

print(f'Linear Model Mean Accuracy: {round(np.mean(lin_acc_list), 3)}%')
print(f'Random Forest Model Mean Accuracy: {round(np.mean(rf_acc_list), 3)}%')


