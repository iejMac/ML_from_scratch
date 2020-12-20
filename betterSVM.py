from matplotlib import pyplot as plt
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


def visualize(data, labels):
    for i, point in enumerate(data):
        if labels[i][0] == -1:
            plt.scatter(point[0], point[1], c='r')
        else:
            plt.scatter(point[0], point[1], c='b')

    x_line = np.linspace(0, 14, 100)
    y_line = [(-1.3*x + 14.6) for x in x_line]
    plt.plot(x_line, y_line)

    plt.show()
    plt.clf()
    return


def get_H(X, y):
    x_prime = y*X
    H = np.dot(x_prime, x_prime.T) * 1.0
    return H


class SVC():
    def __init__(self, kernel=None, degree=None):
        # Add kernel init
        self.w = None
        self.b = None
        self.kernel = kernel
        self.degree = degree

    def fit(self, X, y):
        # Getting all of the parameters for QP
        # 1/2(xT)Px (qT)x  s.t Gx<=h, Ax = b
        # 1/2(aT)Ha - (1T)a s.t (-1)a<=0, (yT)a = 0
        m, n = X.shape

        if self.kernel is not None:
            print(self.kernel)

        H = get_H(X, y)
        P = cvxopt_matrix(H)
        q = cvxopt_matrix(np.ones((m, 1))*-1)
        G = cvxopt_matrix(-np.eye(m))
        h = cvxopt_matrix(np.zeros(m))
        A = cvxopt_matrix(y.T*1.0)
        b = cvxopt_matrix(np.zeros(1))

        cvxopt_solvers.options['show_progress'] = False

        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
        self.w = (np.dot((y*alphas).T, X)).reshape(-1, 1)
        # Finding the support vectors (non-zero alphas)
        S = (alphas > 1e-4).flatten()
        self.b = (y[S] - np.dot(X[S], self.w))[0]

    def predict(self, X):
        # Predicts a singular data point
        decision = np.sign(np.dot(X, self.w) + self.b)
        return decision


X = np.array([[0, 1],
              [1, 2],
              [1, 3],
              [2, 1],
              [3, -1],
              [10, 10],
              [11, 10],
              [12, 13],
              [11, 9],
              [13, 10]])
y = np.array([[-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]]).T
#visualize(X, y)
clf = SVC()
clf.fit(X, y)
