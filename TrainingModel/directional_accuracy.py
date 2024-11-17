from utils.data_handling import sequential_split, load_data
from utils.visualization import plot_split, comp_table
from utils.feature_creation import create_features
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
   

file_path = '../data/BRK.csv'

df = load_data(file_path)

features = create_features(df, classifier=False)

X = features.drop(columns=['Close', 'Date'])
y = features['Close']

split_data = sequential_split(X, y, features['Date'], test_size=0.2)

plot_split(split_data)

X_train = split_data['X_train']
X_test = split_data['X_test']
y_train = split_data['y_train']
y_test = split_data['y_test']

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

predicted_direction = (np.diff(y_pred) > 0).astype(int)
actual_direction = (np.diff(y_test) > 0).astype(int)

dir_acc = accuracy_score(actual_direction, predicted_direction) * 100

r2 = r2_score(y_test, y_pred)

print(f'Directional Accuracy for Linear Regression: {dir_acc:.2f}%')

print(f'R2 Score: {r2}')

comp_table(y_test, y_pred, 'BRK')
print()


# Although the R2 score is much worse for RF, the directional accuracy is slightly better
rfmodel = RandomForestRegressor(n_estimators=1000, max_depth=10, random_state=42)

rfmodel.fit(X_train, y_train)

y_pred = rfmodel.predict(X_test)

r2 = r2_score(y_test, y_pred)

predicted_direction = (np.diff(y_pred) > 0).astype(int)
actual_direction = (np.diff(y_test) > 0).astype(int)

dir_acc = accuracy_score(actual_direction, predicted_direction) * 100

print(f'Directional Accuracy for Random Forest Regression: {dir_acc:.2f}%')

print(f'R2 Score: {r2}')

comp_table(y_test, y_pred, 'BRK')
print()
