import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data preprocessing
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(x)
#testing the simple linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
#Predicting the Test set results
y_pred = lin_reg.predict([[10]])
print(y_pred)
#building the polynomial linear regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=7)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#visualizing the simple linear regression
plt.scatter(x,y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Salary vs Position (simple linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#that plot shows it's not the best fit to the dataset

#now let's plot the polynomial linear regression
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(x_poly), color='blue')
plt.title('Salary vs Position (Polynomial linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#the degree argument at line 15 was changed to 7 to better predict our data (is 4 in course video)


# Visualising the Polynomial Regression results (for higher resolution and smoother curve) (not by me)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
print(lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))