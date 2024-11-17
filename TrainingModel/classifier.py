from sklearn.metrics import accuracy_score
from utils.data_handling import sequential_split, load_data
from utils.feature_creation import create_features
from sklearn.ensemble import RandomForestClassifier
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
    

path = "../data/BRK.csv"

df = load_data(path)

# Need to reverse if using VOO data because recent dates come first in the dataset
#df = df.sort_values(by=['Date']) 

features = create_features(df, classifier=True)

X = features.drop(['Up/Down', 'Date'], axis=1)
y = features['Up/Down']

rf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42)

df = sequential_split(X, y, features['Date'], test_size=0.2 )

rf.fit(df['X_train'], df['y_train'])

y_pred = rf.predict(df['X_test'])

acc = (accuracy_score(df['y_test'], y_pred))*100

print(f'Acc: {acc:.3}%')