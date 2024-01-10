import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# Load local data for regression
regression_data = pd.read_csv('regression_data.csv')
X_reg = regression_data.iloc[:, 0].values.reshape(-1, 1)
y_reg = regression_data.iloc[:, 1].values

# Linear Regression
reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)
y_reg_pred = reg_model.predict(X_reg)

# Visualize the regression data and model
plt.scatter(X_reg, y_reg, label='Data Points', color='blue')
plt.plot(X_reg, y_reg_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linear Regression')
plt.show()

# Load local data for classification
classification_data = pd.read_csv('classification_data.csv')
X_cls = classification_data.iloc[:, :-1].values
y_cls = classification_data.iloc[:, -1].values

# Classification
X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
cls_model = LogisticRegression()
cls_model.fit(X_train, y_train)
y_pred = cls_model.predict(X_test)

# Classification Results
accuracy = accuracy_score(y_test, y_pred)
print(f'Classification Accuracy: {accuracy * 100:.2f}%')

# Visualize the classification results
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Classification Results')
plt.show()
