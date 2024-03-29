import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha*x)

def tanh(x):
    return np.tanh(x)

# Generate x values
x = np.linspace(-5, 5, 100)

# Calculate y values for each activation function
y_sigmoid = sigmoid(x)



y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_tanh = tanh(x)

# Plot the graphs
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, y_relu, label='ReLU')
plt.title('ReLU Activation Function')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.title('Leaky ReLU Activation Function')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, y_tanh, label='Tanh')
plt.title('Tanh Activation Function')
plt.legend()

plt.tight_layout()
plt.show()

