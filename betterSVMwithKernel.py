from matplotlib import pyplot as plt
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.patches as mpatches


def visualize(data, labels, func, test=None):
    for i, point in enumerate(data):
        if labels[i][0] == -1:
            plt.scatter(point[0], point[1], c='r')
        else:
            plt.scatter(point[0], point[1], c='b')

    for point in test:
        plt.scatter(point[0], point[1], c='g')

    # For parabola kernel test:
    if func == 'parabola':
        x1 = np.linspace(0, 10, 100)
        y1 = x1**2 - 10*x1 + 25

    # For 3rd degree kernel test:
    if func == '3rd degree':
        x1 = np.linspace(-3, 3, 100)
        y1 = x1**3

    plt.plot(x1, y1)

    red_patch = mpatches.Patch(color='red', label='False')
    blue_patch = mpatches.Patch(color='blue', label='True')
    green_patch = mpatches.Patch(color='green', label='Test')
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc=9)

    plt.show()
    plt.clf()
    return


def get_H(X, y):
    x_prime = y*X
    H = np.dot(x_prime, x_prime.T) * 1.0
    return H


def get_kernel(x1, x2, kernel, degree):
    transformation = None
    if kernel == 'polynomial':
        transformation = np.power(np.dot(x1, x2.T) + 1, degree)
    return transformation


class SVC:
    def __init__(self, kernel=None, degree=None):
        self.w = None
        self.b = None
        # alphas * y
        self.coef = None
        self.kernel = kernel
        self.degree = degree
        self.support_vecs = None
        self.sv_types = None

    def fit(self, X, y):
        # Getting all of the parameters for QP
        # 1/2(xT)Px (qT)x  s.t Gx<=h, Ax = b
        # 1/2(aT)Ha - (1T)a s.t (-1)a<=0, (yT)a = 0

        m, n = X.shape
        y_temp = y.T[0]

        if self.kernel is 'polynomial':
            K = get_kernel(X, X, self.kernel, self.degree)
            H = np.dot(y, y.T) * K * 1.0
        else:
            H = get_H(X, y)

        P = cvxopt_matrix(H)
        q = cvxopt_matrix(np.ones((m, 1))*-1)
        G = cvxopt_matrix(-np.eye(m))
        h = cvxopt_matrix(np.zeros(m))
        A = cvxopt_matrix(y_temp.reshape(1, -1)*1.0)
        b = cvxopt_matrix(np.zeros(1))

        cvxopt_solvers.options['show_progress'] = False
        cvxopt_solvers.options['abstol'] = 1e-10
        cvxopt_solvers.options['reltol'] = 1e-10
        cvxopt_solvers.options['feastol'] = 1e-10

        # Had a BIG problem when i scaled up the numbers
        # ValueError: Rank(A) < p or Rank(P; G; A) < n
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)

        alphas = np.array(sol['x'])

        S = (alphas > 1e-4).flatten()
        self.support_vecs = X[S]
        self.sv_types = y[S]

        if self.kernel is 'polynomial':
            self.coef = alphas[S] * y[S]
        else:
            self.w = (np.dot((y*alphas).T, X)).reshape(-1, 1)
            # Finding the support vectors (non-zero alphas)
            self.b = (y[S] - np.dot(X[S], self.w))[0]

    def predict(self, X):
        if self.kernel is 'polynomial':
            any_support_vector = self.support_vecs[0]
            ym = self.sv_types[0]

            kernel = get_kernel(X, self.support_vecs, self.kernel, self.degree)

            b_kernel = get_kernel(self.support_vecs, any_support_vector, self.kernel, self.degree)
            self.b = ym - np.dot(b_kernel, self.coef)

            decision = np.sign(np.dot(kernel, self.coef) + self.b)
            return decision
        else:
            decision = np.sign(np.dot(X, self.w) + self.b)
        return decision


X_train_parabola = np.array([[0, 1],
                          [1, 2],
                          [1, 3],
                          [2, 1],
                          [3, -1],
                          [10, 10],
                          [11, 10],
                          [12, 13],
                          [11, 9],
                          [13, 10],
                          [5, 10],
                          [4, 7],
                          [6, 4],
                          [8, 15],
                          [2, 20],
                          [10, 0]])
y_train_parabola = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1]]).T

X_test_parabola = np.array([[6, 15],
                   [8, 0],
                   [12, 20],
                   [12, 0],
                   [0, 0]])

y_test_parabola = np.array([[1, -1, -1, -1]]).T

X_train_3rd_deg = np.array([[0, 2],
                        [2, 10],
                        [-2, 5],
                        [-1, 0],
                        [-3, 4],
                        [-1, 6],
                        [-1, 15],
                        [1, 20],
                        [0, -4],
                        [1, -2],
                        [2, 5],
                        [2, 3],
                        [3, 0],
                        [1, -6],
                        [2, -3],
                        [3, -10],
                        [1, -1]])

y_train_3rd_deg = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).T

X_test_3rd_deg = np.array([[2, 16],
                           [-2, -9],
                           [0, 3],
                           [1, -1],
                           [-2, 0]])

y_test_3rd_deg = np.array([[-1, 1, -1, 1, -1]]).T


# Testing data divided by quadratic func.
# ----------------------------------------
clf = SVC(kernel='polynomial', degree=2)
clf.fit(X_train_parabola, y_train_parabola)
predictions = clf.predict(X_test_parabola)
print(predictions == 1)

visualize(X_train_parabola, y_train_parabola, 'parabola', X_test_parabola)

# Testing data divided by 3rd degree polynomial
# ---------------------------------------------
# clf = SVC(kernel='polynomial', degree=3)
# clf.fit(X_train_3rd_deg, y_train_3rd_deg)
# predictions = clf.predict(X_test_3rd_deg)
# print(predictions == 1)
#
# visualize(X_train_3rd_deg, y_train_3rd_deg, '3rd degree', X_test_3rd_deg)

# ==============================================================================
# Weaknesses to work on:
# - Breaks when the training samples have big values
# - Would be easier to troubleshoot if I knew what was happening in cvxopt

