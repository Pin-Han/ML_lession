# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# print('X: ->', X)
# print('y: ->', y)  

# Training the Decision Tree Regression model on the whole dataset

# random_state=0, 讓 Model 每次跑的結果都一樣
regressor = DecisionTreeRegressor(random_state=0) 

regressor.fit(X, y)

# Predicting a new result
regressor.predict([[6.5]])

# Visualizing the Decision Tree Regression results (higher resolution)
x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()