import numpy as np
from matplotlib import pyplot as plt


class SVMclf:
    def __init__(self, epochs, eta):
        self.epochs = epochs
        self.eta = eta

    def fit(self, X, y):
        reg = 1/self.epochs
        w = np.zeros((len(X[0])))
        for i in range(self.epochs):

            for j, x in enumerate(X):
                # If misclassified
                if y[j]*np.dot(w, X[j]) < 1:
                    # Adjust weights accordingly
                    w = w + self.eta*(X[j]*y[j] + (-2 * reg * w))
                else:
                    w = w + self.eta*(-2 * reg * w)
        self.w = w

    def predict(self, X):
        pass


def show_data(all_data, params):
    rows = all_data.T.copy()
    y = rows[2]
    class_1_ind = np.where(y == 1)
    class_2_ind = np.where(y == 0)
    x1_c1 = rows[0][class_1_ind]
    x2_c1 = rows[1][class_1_ind]
    x1_c2 = rows[0][class_2_ind]
    x2_c2 = rows[1][class_2_ind]
    plt.scatter(x1_c1, x2_c1, c='red')
    plt.scatter(x1_c2, x2_c2, c='blue')

    x_line = np.linspace(0, 20, 100)
    y_line = params[1]*x_line + params[0]
    plt.plot(x_line, y_line, color='green')

    plt.show()
    plt.clf()
    return


x1 = np.array([1, 2, 4, 5, 6, 10, 11, 13, 15, 16])
x2 = np.array([10, 9, 8, 10, 11, 1, 2, 4, 5, 3])
X = np.c_[x1, x2]
y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

data = np.c_[x1, x2, y]

svc = SVMclf(epochs=1000, eta=5)
svc.fit(X, y)
print(svc.w)

show_data(data, svc.w)

