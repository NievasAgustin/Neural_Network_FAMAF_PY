import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load your dataset into a pandas DataFrame
# For this example, we'll use a sample dataset.
# Replace 'your_dataset.csv' with the path to your dataset.
data = pd.read_csv('C:\\Users\\agust\\Documents\\Nueva Compu\\2023\\Doctorado\\Before\\path_to_your_dataset.csv')

# Explore and visualize your data (optional)
# sns.pairplot(data)  # You can use this to create scatter plots
# plt.show()

# Select your feature(s) and target variable
X = data[['Feature']]  # Replace 'Feature' with the actual column name
y = data['Target']  # Replace 'Target' with the actual column name

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to your training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance (for regression problems)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the data points
plt.scatter(X, y, color='blue', label='Data Points')

# Plot the linear regression line
plt.plot(X, model.predict(X), color='red', linewidth=3, label='Linear Regression Line')

plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Linear Regression')
plt.legend()
plt.show()
