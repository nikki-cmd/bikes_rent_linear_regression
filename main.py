from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt


path = "dataset/bikes_rent.csv"

df = pd.read_csv(path)
dataframe = pd.DataFrame(df)
print(dataframe.columns)
dataframe = dataframe.drop(labels=['season', 'atemp', 'windspeed(mph)'], axis=1)
train, test = train_test_split(dataframe, test_size=0.2)
X = train.drop(['cnt'], axis=1)
y = train.drop(['yr', 'mnth', 'holiday', 'weekday', 'workingday',
       'weathersit', 'temp', 'hum', 'windspeed(ms)'], axis=1)

correlation_matrix = dataframe.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()




reg = LinearRegression().fit(X, y)

test = test.drop(['cnt'], axis=1)

prediction = reg.predict(test)

print(prediction)

