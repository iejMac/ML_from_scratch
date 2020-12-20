import numpy as np

np.random.seed(42)

X = np.array([[1, 0, 0, 0, 1, 1, 1, 1],
             [0, 1, 0, 1, 0, 1, 0, 1],
             [0, 0, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 1, 0, 1, 1],
             [1, 1, 1, 1, 0, 0, 0, 1],
             [1, 1, 0, 1, 1, 1, 0, 0],
             [0, 1, 1, 0, 0, 0, 0, 0],
             [1, 0, 1, 0, 0, 1, 1, 0]],
             np.int32
             )

y = np.array([[143, 85, 48, 27, 241, 220, 96, 166]]).T

# Initializing weights close to 0:
weights = 2*np.random.rand(8, 1) - 1

# Hidden layer:
#h_layer = 2*np.random.rand(2, 1) - 1
# No hidden layer for now, we want weights to be (8, 4, 2, 1)


eta = 0.01
n_epochs = 10000

for epoch in range(n_epochs):
    # Forward prop.
    y_pred = np.zeros((8, 1))
    papw = np.zeros((8, 1))
    for i in range(8):
        y_pred[i] = np.dot(X[i], weights)
        papw[i] = X[i].sum()
    pCpa = (y_pred - y)
    error_delta = pCpa * papw
    adjustment = eta * np.dot(X.T, error_delta)
    weights -= adjustment

print(weights)
