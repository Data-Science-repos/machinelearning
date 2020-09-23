import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

# feature scaling, the dependant variable has to be also scaled( using a new standard_scaler object because the first
# object would use the information obtained from scaling x

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x = sc_x.fit_transform(x)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)


# Training the SVR model
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(x, y)

# Prediction and Reversing scaling to see the unscaled salaries
print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))))

# visualizing the model
print(x)
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x)), color='magenta')
plt.title('Truth or Lie (SVR Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Visualising the Support Vector Regression results (for higher resolution and smoother curve) (not by me)
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color='blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()





