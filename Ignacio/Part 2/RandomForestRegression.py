import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Data Set

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10, random_state= 0)
regressor.fit(x,y)

# Predicting a new result
print(regressor.predict([[6.5]]))

# Visualising the Random Forest Regression results (higher resolution) (but a 2d graph is not the best way to visualize it)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


