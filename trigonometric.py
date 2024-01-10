import numpy as np
import matplotlib.pyplot as plt

# Create an array of x values
x = np.linspace(0, 2 * np.pi, 100)

# Define phase shifts for the three sine waves
phases = [0, 120, 240]

# Initialize the plot
fig, ax = plt.subplots()

# Plot three sine waves with different phase shifts
for phase in phases:
    sine_wave = np.sin(x + np.deg2rad(phase))
    ax.plot(x, sine_wave, label=f'Sin({phase}°)')

# Summing up the three sine waves
summed_wave = np.sin(x) + np.sin(x + np.deg2rad(120)) + np.sin(x + np.deg2rad(240))
ax.plot(x, summed_wave, label='Sum')

# Show the legend
ax.legend()

plt.show()
