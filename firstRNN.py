import numpy as np


def softmax(x):
    x = x - x.max()
    y = np.exp(x)
    return y / y.sum()


def d_softmax(x):
    return softmax(x)*(1-softmax(x))


def forward_pass(input_neurons, previous_hidden, neuron_weights, neuron_biases):
    a_t = neuron_weights[0] @ input_neurons + neuron_weights[1] @ previous_hidden + neuron_biases[0]
    h_t = np.tanh(a_t)
    b_t = neuron_weights[2] @ h_t + neuron_biases[1]
    y_t = softmax(b_t)
    return a_t, b_t, h_t, y_t


def full_forward_pass(x, h0, all_weights, all_biases):
    b_t = []
    a_t = []
    h_t = [h0]
    y_t = []
    for i, letter in enumerate(x):
        current_a, current_b, current_h, current_y = forward_pass(letter, h_t[i], all_weights, all_biases)
        a_t.append(current_a)
        b_t.append(current_b)
        h_t.append(current_h)
        y_t.append(current_y)

    return np.array(a_t), np.array(b_t), np.array(h_t), np.array(y_t)


def full_backward_pass(a_t, b_t, h_t, y_t, x_t, y_at, n_weights, n_biases):

    dLdWhh = np.zeros(n_weights[1].shape)
    dLdbhh = np.zeros(n_biases[0].shape)

    dLdWhy = np.zeros(n_weights[2].shape)
    dLdbhy = np.zeros(n_biases[1].shape)

    dLdWxh = np.zeros(n_weights[0].shape)

    for i in reversed(range(len(y_t))):

        # y:
        dL_dWhy, dL_dbhy = d_L_d_Why_bhy(y_t[i], y_at[i], h_t[i])
        dLdWhy += dL_dWhy
        dLdbhy += dL_dbhy

        d_y = dL_dbhy.copy()
        d_h = n_weights[2].T @ d_y

        for j in range(i + 1):
            temp = d_tanh(a_t[i-j]) * d_h
            dLdbhh += temp
            dLdWhh += temp @ h_t[i-j].T

            dLdWxh += temp @ x_t[i-j].T

            d_h = n_weights[1] @ temp

    for d in [dLdWxh, dLdWhh, dLdWhy, dLdbhh, dLdbhy]:
        np.clip(d, -1, 1, out=d)

    return [dLdWxh, dLdWhh, dLdWhy], [dLdbhh, dLdbhy]


def categorical_cross_entropy(y_prob, y_actual):
    return -np.sum(y_actual * np.log(y_prob))


def d_cce(y_prob, y_actual):
    return -np.sum(y_actual/y_prob)


def d_tanh(inp):
    return (1 - np.tanh(inp)**2)[0]


def d_L_d_Why_bhy(y_pred, y_actual, h_t):

    ind = np.argmax(y_actual)
    dLdby = y_pred
    dLdby[ind] -= 1

    dLdWhy = dLdby @ h_t.T

    return dLdWhy, dLdby


def update_weights(weights, biases, w_gradients, b_gradients):
    new_weights = []
    new_biases = []
    learning_rate = 0.01
    for i, weight in enumerate(weights):
        new_weights.append(weights[i] - learning_rate*w_gradients[i])
    for i, bias in enumerate(biases):
        new_biases.append(biases[i] - learning_rate*b_gradients[i])
    return new_weights, new_biases


def exposition(x, h_0, n_weights, n_biases, n_letters):
    h_prev = h_0
    current_x = x
    for i in range(n_letters):
        ind = np.argmax(current_x)
        if ind == 0:
            print('h')
        elif ind == 1:
            print('e')
        elif ind == 2:
            print('l')
        elif ind == 3:
            print('o')
        a, b, h, y = forward_pass(current_x, h_prev, n_weights, n_biases)
        current_x = y
        h_prev = h

# Notes:
# Teach model to predict the word "hello"

word = 'hello'

vocab = ['h', 'e', 'l', 'o']

h = np.array([1, 0, 0, 0]).reshape((4, 1))
e = np.array([0, 1, 0, 0]).reshape((4, 1))
l = np.array([0, 0, 1, 0]).reshape((4, 1))
o = np.array([0, 0, 0, 1]).reshape((4, 1))

# Data:
X = np.array([h, e, l, l])
y = np.array([e, l, l, o])

# Hyperparameters:
hidden_units = 120
input_units = len(vocab)
output_units = len(vocab)
h_0 = np.zeros((hidden_units, 1))

# Initializing weight matrices
Wxh = np.random.rand(hidden_units, input_units)*2 - 1
Whh = np.random.rand(hidden_units, hidden_units)*2 - 1
bhh = np.zeros((hidden_units, 1))
Why = np.random.rand(output_units, hidden_units)*2 - 1
bhy = np.zeros((output_units, 1))
weights = [Wxh, Whh, Why]
biases = [bhh, bhy]

epochs = 600

for i in range(epochs):
    a_list, b_list, h_list, y_list = full_forward_pass(X, h_0, weights, biases)
    w_gradients, b_gradients = full_backward_pass(a_list, b_list, h_list, y_list, X, y, weights, biases)
    weights, biases = update_weights(weights, biases, w_gradients, b_gradients)


exposition(h, h_0, weights, biases, 5)

