import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

filename = "./11/09_irisdata.csv"

column_names = ['sepal-length', 'sepal-width', 'petal-length' 'petal-width', 'class']

data = pd.read_csv(filename, names=column_names)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

kfold = KFold(n_splits=10, random_state=5, shuffle=True)
model = DecisionTreeClassifier()
results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
print("데이터 셋의 행렬 크기(shape):", data.shape)
print("데이터 셋의 요약(describe()):")
print(data.describe())
class_counts = data.groupby('class').size()
print("데이터 셋의 클래스 종류:\n", class_counts)

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

data = pd.read_csv(filename, names=column_names)

scatter_matrix(data)
plt.savefig("./11/scatter_plot.png")
