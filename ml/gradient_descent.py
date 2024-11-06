# %% [markdown]
# Implement Gradient Descent Algorithm to find the local minima of a function.
# For example, find the local minima of the function y=(x+3)² starting from the point x=2

# %%
# Define the function
def function_to_minimize(x):
    return (x + 3)**2

# %%
# Define the derivative of the function
def derivative(x):
    return 2 * (x + 3)

# %%
# Gradient Descent Algorithm
gd=[]
def gradient_descent(starting_x, learning_rate, num_iterations):
    x = starting_x
    for i in range(num_iterations):
        gradient = derivative(x)
        gd.append(x)
        x = x - learning_rate * gradient
        # You can print the value of x at each iteration to see the progress
        print(f"Iteration {i + 1}: x = {x}, f(x) = {function_to_minimize(x)}")
    return x

# %%

# Initial parameters
initial_x = 2
learning_rate = 0.1
iterations = 100

# Find the local minimum
minima = gradient_descent(initial_x, learning_rate, iterations)

print(f"Local minimum occurs at x = {minima}, f(x) = {function_to_minimize(minima)}")

# %%
import seaborn as sns

sns.scatterplot(gd)

# %%
!pip install sympy
import sympy as sp

# %%
# Define the function to minimize
def function_to_minimize(x):
    return x**3 - 6*x**2 + 11*x - 6  # Example function: x^3 - 6x^2 + 11x - 6

# %%
# Calculate the derivative symbolically using SymPy
x = sp.symbols('x')
derivative = sp.diff(function_to_minimize(x), x)
derivative_function = sp.lambdify(x, derivative, 'numpy')

# %%
# Gradient Descent Algorithm
def gradient_descent(starting_x, learning_rate, num_iterations, derivative_fn):
    x = starting_x
    for i in range(num_iterations):
        gradient = derivative_fn(x)
        x = x - learning_rate * gradient
        # You can print the value of x at each iteration to see the progress
        print(f"Iteration {i + 1}: x = {x}, f(x) = {function_to_minimize(x)}")
    return x

# %%
# Initial parameters
initial_x = 2  # Initial starting point
learning_rate = 0.1  # Adjust this based on your function and needs
iterations = 100  # Adjust the number of iterations as needed

# Find the local minimum
minima = gradient_descent(initial_x, learning_rate, iterations, derivative_function)

print(f"Local minimum occurs at x = {minima}, f(x) = {function_to_minimize(minima)}")

# %%
import math

# Define the function to minimize
def function_to_minimize(x):
    return x**3 - 6*x**2 + 11*x - 6  # Example function: x^3 - 6x^2 + 11x - 6

# %%
# Calculate the derivative using numerical differentiation
def derivative(x, epsilon=1e-6):
    return (function_to_minimize(x + epsilon) - function_to_minimize(x)) / epsilon

# %%
# Gradient Descent Algorithm
def gradient_descent(starting_x, learning_rate, num_iterations, derivative_fn):
    x = starting_x
    for i in range(num_iterations):
        gradient = derivative_fn(x)
        x = x - learning_rate * gradient
        # You can print the value of x at each iteration to see the progress
        print(f"Iteration {i + 1}: x = {x}, f(x) = {function_to_minimize(x)}")
    return x

# %%
# Initial parameters
initial_x = 2  # Initial starting point
learning_rate = 0.1  # Adjust this based on your function and needs
iterations = 100  # Adjust the number of iterations as needed

# Find the local minimum
minima = gradient_descent(initial_x, learning_rate, iterations, derivative)

print(f"Local minimum occurs at x = {minima}, f(x) = {function_to_minimize(minima)}")

# %%



