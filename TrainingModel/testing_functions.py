from utils.data_handling import sequential_split, load_data
from utils.visualization import plot_split

file_path = './data/BRK.csv'
features = ['Open', 'High', 'Low', 'Volume']
target = ['Close']

X, y, dates = load_data(file_path, features, target)

split_data = sequential_split(X, y, dates, test_size=0.2)

plot_split(split_data)