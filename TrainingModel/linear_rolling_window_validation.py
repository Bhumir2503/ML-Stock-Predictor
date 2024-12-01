import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from utils.data_handling import sequential_split, load_data
from utils.feature_creation import create_features
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
    

path = "../data/BRK.csv"

df = load_data(path)

features = create_features(df, classifier=False)
features = features.dropna()

X = features.drop(columns=['Close', 'Date'])
y = features['Close']

lin_model = LinearRegression()
rfmodel = RandomForestRegressor(n_estimators=1000, max_depth=10, random_state=42)
knn_model = KNeighborsRegressor(n_neighbors = 5)

fold_length = np.floor(len(X)/10).astype(int)   #Assigns 220 to fold_length (roughly 10% of dataset)
training_length = (np.floor(len(X)/10)*5).astype(int)  #Assigns 1100 to training_length (roughly 50% of dataset)

lin_acc_list = []   #Empty list to store directional accuracies of linear model
rf_acc_list = []    #Empty list to store directional accuracies of random forest model
knn_acc_list = []   #Empty list to store directional accuracies of knn model

'''Perform a rolling window cross validation where initial 50% of data is used for training and 
   the next 10% is used for testing. On iteration 2, data range from 10%-60% is used for training
   and range from 60% - 70% is used for testing, and so on.. '''
i = 0
for j in range(1, 6):
    training_subset_X = X[(i*fold_length):training_length+(i*fold_length)]
    training_subset_y = y[(i*fold_length):training_length+(i*fold_length)]
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

    '''Fitting and testing 5 different data subsets for the knn model'''
    knn_model.fit(training_subset_X, training_subset_y)
    validation_set_predictions = knn_model.predict(testing_subset_X)
    predicted_direction = (np.diff(validation_set_predictions) > 0).astype(int)
    actual_direction = (np.diff(testing_subset_y) > 0).astype(int)
    dir_acc = accuracy_score(actual_direction, predicted_direction) * 100
    #print(dir_acc)
    knn_acc_list.append(dir_acc)

    i+=1

print(f'Linear Model Mean Directional Accuracy: {round(np.mean(lin_acc_list), 3)}%')
print(f'Random Forest Model Mean Directional Accuracy: {round(np.mean(rf_acc_list), 3)}%')
print(f'KNN Model Mean Directional Accuracy: {round(np.mean(knn_acc_list), 3)}%')


from scipy.stats import t

confidence_interval = 0.95
folds = 5
degrees_freedom = folds - 1
t_val = t.ppf((1 + confidence_interval) / 2, degrees_freedom)

stand_dev_lin = np.std(lin_acc_list)
stand_dev_rf = np.std(rf_acc_list)
stand_dev_knn = np.std(knn_acc_list)

sem_lin = stand_dev_lin / np.sqrt(folds)
sem_rf = stand_dev_rf / np.sqrt(folds)
sem_knn = stand_dev_knn / np.sqrt(folds)

ci_margin_lin = t_val * sem_lin
ci_margin_rf = t_val * sem_rf
ci_margin_knn = t_val * sem_knn

print(f'Linear Model Mean Directional Accuracy: {round(np.mean(lin_acc_list), 3)}%')
print(f'\t Lower bound (95% CI): {round(np.mean(lin_acc_list) - ci_margin_lin, 3)}%')
print(f'\t Upper bound (95% CI): {round(np.mean(lin_acc_list) + ci_margin_lin, 3)}%')

print(f'Random Forest Model Mean Directional Accuracy: {round(np.mean(rf_acc_list), 3)}%')
print(f'\t Lower bound (95% CI): {round(np.mean(rf_acc_list) - ci_margin_rf, 3)}%')
print(f'\t Upper bound (95% CI): {round(np.mean(rf_acc_list) + ci_margin_rf, 3)}%')

print(f'KNN Model Mean Directional Accuracy: {round(np.mean(knn_acc_list), 3)}%')
print(f'\t Lower bound (95% CI): {round(np.mean(knn_acc_list) - ci_margin_knn, 3)}%')
print(f'\t Upper bound (95% CI): {round(np.mean(knn_acc_list) + ci_margin_knn, 3)}%')

