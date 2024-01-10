import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read the data from 'Exceldata.txt' (assuming it's a single-column dataset)
data = pd.read_csv('Exceldata.txt', delimiter='\t', header=None, names=['Y'])

# Create an implicit 'X' column representing data point indices
data['X'] = range(1, len(data) + 1)

# Extract the 'X' and 'Y' values
X = data[['X']].values
Y = data['Y'].values

# Linear Regression
reg_model = LinearRegression()
reg_model.fit(X, Y)
Y_pred = reg_model.predict(X)

# Visualize the regression results
plt.scatter(X, Y, label='Data Points', color='blue')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('X (Data Point Index)')
plt.ylabel('Y')
plt.legend()
plt.title('Linear Regression')
plt.show()
