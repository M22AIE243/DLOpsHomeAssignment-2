import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
sigmoid_values = sigmoid(np.array(random_values))

print("Sigmoid values for the given data:")
for value, sigmoid_value in zip(random_values, sigmoid_values):
    print(f"Sigmoid({value}) = {sigmoid_value}")
