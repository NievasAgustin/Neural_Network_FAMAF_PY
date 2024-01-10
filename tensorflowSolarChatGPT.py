import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load your data from the text file
data = pd.read_csv('Exceldata.txt', header=None)

# Extract the single column as your input data
X = data.iloc[:, 0]

# Create corresponding output data (you need to specify this)
# For simplicity, let's assume the output is just the input squared
y = X**2

# Build a simple feedforward neural network
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)  # Output layer with a single neuron for regression
])

# Compile the model with a loss function and optimizer
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X, y, epochs=1000, verbose=0)

# Plot the training loss
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Generate predictions
y_pred = model.predict(X)

# Plot the original data and model predictions
plt.scatter(X, y, label='Original Data')
plt.plot(X, y_pred, label='Predictions', linestyle='dashed')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Function Approximation')
plt.show()
