import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_to_derivative(output):
    return output*(1-output)


# Input data
X = np.array([
    [1, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1]
])

# Desired output vector
y = np.array([[0, 1, 1, 0]]).T

# Setting random seed
np.random.seed(1)

# Initializing first weights at random
synapse_0 = 2*np.random.random((4, 4)) - 1
synapse_1 = 2*np.random.random((4, 1)) - 1
activation_2 = np.zeros((4, 1))

for i in range(100000):
    # Calculating the weighted sum of the input layer with the weight vector
    # FeedForward:
    activation_0 = X
    activation_1 = sigmoid(np.dot(activation_0, synapse_0))
    activation_2 = sigmoid(np.dot(activation_1, synapse_1))

    layer_2_error = activation_2 - y
    layer_2_delta = layer_2_error * sigmoid_to_derivative(activation_2)

    layer_1_error = layer_2_delta.dot(synapse_1.T)
    layer_1_delta = layer_1_error * sigmoid_to_derivative(activation_1)

    cost_gradient_2 = activation_1.T.dot(layer_2_delta)
    cost_gradient_1 = activation_2.T.dot(layer_1_delta)

    synapse_0 -= 1*cost_gradient_1
    synapse_1 -= 1*cost_gradient_2

print(activation_2)

