import numpy as np
from sympy import symbols, Matrix, I, sqrt, simplify, Abs, lambdify

# Define the symbolic variables for the density matrix elements
a, b, c, d, e, f = symbols('a b c d e f', real=True)

# Define your density matrix 'rho' using the symbolic variables
rho = Matrix([[a, b, 0, 0], [b.conjugate(), c, 0, 0], [0, 0, d, e], [0, 0, e.conjugate(), f]])

# Calculate the spin-flipped density matrix
rho_star = Matrix([[a, b, 0, 0], [b.conjugate(), c, 0, 0], [0, 0, d, e], [0, 0, e.conjugate(), f]])

# Compute the matrix C
C = (rho * rho_star).applyfunc(simplify)

# Calculate the eigenvalues of C
eigenvalues = C.eigenvals(multiple=True)
eigenvalues_magnitudes = [Abs(eigenval) for eigenval in eigenvalues]

print("Magnitudes of eigenvalues:", eigenvalues_magnitudes)

# Create a lambda function to convert SymPy expressions to numerical values
eigenvalue_function = lambdify((a, b, c, d, e, f), eigenvalues_magnitudes, modules=['numpy'])

print("Eigenvalues function:", eigenvalue_function)

# Evaluate the eigenvalues (provide the values for a, b, c, d, e, and f)
numerical_eigenvalues = eigenvalue_function(0.2, 0.2, 0.2, 0.2, 0.2, 0.2)  # Replace with the actual values

# Sort the eigenvalues in decreasing order of magnitude
eigenvalues_sorted = sorted(numerical_eigenvalues, reverse=True)

# Calculate the concurrence
concurrence = max(0, 2 * sqrt(eigenvalues_sorted[0]) - sqrt(eigenvalues_sorted[1]) - sqrt(eigenvalues_sorted[2]) - sqrt(eigenvalues_sorted[3]))

print("Eigenvalues of C:", numerical_eigenvalues)
print("Concurrence:", concurrence)
