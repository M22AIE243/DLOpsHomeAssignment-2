import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

sigmoid_values = sigmoid(np.array(random_values))
relu_values = relu(np.array(random_values))
leaky_relu_values = leaky_relu(np.array(random_values))
tanh_values = tanh(np.array(random_values))

print("Sigmoid values for the given data:")
for value, sigmoid_value in zip(random_values, sigmoid_values):
    print(f"Sigmoid({value}) = {sigmoid_value}")

print("\nReLU values for the given data:")
for value, relu_value in zip(random_values, relu_values):
    print(f"ReLU({value}) = {relu_value}")

print("\nLeaky ReLU values for the given data:")
for value, leaky_relu_value in zip(random_values, leaky_relu_values):
    print(f"Leaky ReLU({value}) = {leaky_relu_value}")

print("\nTanh values for the given data:")
for value, tanh_value in zip(random_values, tanh_values):
    print(f"Tanh({value}) = {tanh_value}")
