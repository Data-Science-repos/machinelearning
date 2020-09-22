import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#Values of 0 seem to be ok in this case (don't know the reason)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#in multiple linear regression there's no need to apply feature scaling because each variable coefficient ( b1x1,b2x2,bNxN) will help bring every variable to the same scale

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 0)

#training

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(x_train)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
