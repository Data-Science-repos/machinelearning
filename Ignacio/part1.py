import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# creating dataset
dataset = pd.read_csv('data.csv')
print(dataset)

# preparing data
# x = matrix of FEATURES
x = dataset.iloc[:, :-1].values
print(x)
# y = DEPENDeNT VARIABLE
y = dataset.iloc[:, -1].values

print(y)

#missing data clean up with average imputer

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)

#encoding categorical data

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)

#encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

#splitting dataset into training set and test set. Always do this before feature scaling so as not to affect the test data (usually 80/20 train/test ratio)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

#next, feature scaling. if data follows a normal distribution then use the normalisation method, otherwise the standardisation methode does a great job all time.
#the standardisation method is mostly used

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
#values from country (which are 1 0 0, 0 1 0 and 0 0 1) dont need to be standardised since they already fit in the 2 to -2 of the standardisation process

#Next, we will find the mean and the standard deviation(using FIT) and 'transform' the x_train values
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])

# now the x_test will only get transformed using the mean and the standard deviation found with the x_train variables since we are not 'supposed' to have the test values

x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)

print(x_test)












