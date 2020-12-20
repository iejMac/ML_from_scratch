import numpy as np
from matplotlib import pyplot as plt

np.random.seed(1)


# Function that makes filters:
def initialize_filters(filter_size, filter_count):
    return np.random.uniform(-1, 1, (filter_count, filter_size, filter_size))


# Function that convolves the image with the filter
def convolve(images, filters, stride):
    image_l, _ = images[0].shape
    filter_l, _ = filters[0].shape
    output_dim = int((image_l - filter_l)/stride) + 1
    outputs = []
    for __ in images:
        for _ in filters:
            outputs.append(np.zeros((output_dim, output_dim)))
    counter = 0
    for y, image in enumerate(images):
        for x, feature_filter in enumerate(filters):
            i = 0
            while filter_l + i <= image_l:
                j = 0
                while filter_l + j <= image_l:
                    outputs[counter][i][j] = np.sum(image[i:i+filter_l, j:j+filter_l] * feature_filter)
                    j += stride
                i += stride
            counter += 1

    return np.asarray(outputs)


# Max pooling function:
def max_pool(images, pool_size, stride):
    image_l, _ = images[0].shape
    output_dim = int((image_l - pool_size)/stride) + 1
    outputs = []
    for _ in images:
        outputs.append(np.zeros((output_dim, output_dim)))

    for x, image in enumerate(images):
        i = 0
        curr_y = 0
        while curr_y + pool_size <= image_l:
            j = 0
            curr_x = 0
            while curr_x + pool_size <= image_l:
                outputs[x][i][j] = np.max(image[curr_y:curr_y+pool_size, curr_x:curr_x+pool_size])
                j += 1
                curr_x += stride
            i += 1
            curr_y += stride

    return np.asarray(outputs)


# Some supplementary functions:
def flatten_layer(images):
    return images.flatten()


def return_dimensions(flat_layer, shape):
    flat_layer = np.asarray(flat_layer)
    return flat_layer.reshape(shape)


def feed_forward(weights, biases, x):
    activation = (weights @ x) + biases
    # Scaling:
    activation = relu(activation)
    return activation


def std_scaler(x):
    std = np.std(x)
    mean = np.mean(x)
    activation = [(unit - mean) / std for unit in x]
    return activation


def softmax(x):
    expo = np.exp(x)
    activation = expo / np.sum(expo)
    return activation


def relu(x):
    z = np.zeros(x.shape)
    activation = np.where(x > 0, x, z)
    return activation


def softmax_grad(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def categorical_cross_entropy(y_prob, y_actual):
    return -np.sum(y_actual * np.log(y_prob))


def rot180(np_array):
    return np.rot90(np_array, 2)


def show_face(grid):
    plt.matshow(grid)
    plt.show()
    plt.clf()


def predict_mood(face, CNN):
    prediction = CNN.predict([face])
    show_face(face)
    ind = np.argmax(prediction)
    if ind == 0:
        print("You're in a good mood :)")
    elif ind == 1:
        print("You're in an ok mood :/")
    elif ind == 2:
        print("You're in a bad mood :(")


# Backward max-pool
def backward_maxpool(grid, repooled, stride, map, gradients):
    for x, image in enumerate(map):
        image_l, _ = image.shape
        pool_l, _ = repooled[0].shape
        i = 0
        curr_y = 0
        while curr_y + pool_l <= image_l:
            j = 0
            curr_x = 0
            while curr_x + pool_l <= image_l:
                sub_matrix = image[curr_y:curr_y+pool_l, curr_x:curr_x+pool_l]
                ind = np.unravel_index(np.argmax(sub_matrix), sub_matrix.shape)
                ind_x, ind_y = ind
                grid[x][ind_y + curr_y][ind_x + curr_x] = gradients[x][i][j]
                j += 1
                curr_x += stride

            i += 1
            curr_y += stride
    return grid


# Backward convolution:
def backward_convolution_wrt_weights(gradients, layers, filter_count, filter_shape, stride):
    update_f1 = np.zeros((filter_shape, filter_shape))
    update_f2 = np.zeros((filter_shape, filter_shape))
    # Fragmenting the layers:
    counter = 0
    for layer in layers:
        image = np.array([layer])
        for i in range(filter_count):
            grads = [np.asarray(gradients[counter])]
            update = convolve(image, grads, stride)
            if i is 0:
                update_f1 += update.reshape((3, 3))
            else:
                update_f2 += update.reshape((3, 3))
            counter += 1
    return update_f1, update_f2


def backward_convolution_wrt_inputs(gradients, filters, layer1_shape, stride):
    image_count = layer1_shape[0]
    image_size = layer1_shape[1]
    filter_count = filters.shape[0]
    gradients_new = np.zeros(layer1_shape)
    counter = 0
    for i in range(image_count):
        for j in range(filter_count):
            image = filters[j]
            image_rot = rot180(image)
            padded = np.pad(image_rot, (3, 3), "constant", constant_values=(0, 0))
            padded_size = padded.shape[0]
            grad = gradients[counter]
            grad_size = grad.shape[0]
            m = 0
            curr_y = padded_size - grad_size
            while curr_y >= 0:
                n = 0
                curr_x = padded_size - grad_size
                while curr_x >= 0:
                    square = padded[curr_y:curr_y+grad_size, curr_x:curr_x+grad_size]
                    gradients_new[i][m][n] += np.sum(square * grad)
                    n += 1
                    curr_x -= stride
                m += 1
                curr_y -= stride

            counter += 1
    return gradients_new


class CNN:
    def __init__(self, filter_size=3, filter_stride=1, filter_count=2, pool_size=2, pool_stride=2,
                 fully1_size=16, fully2_size=16, output_size=3, learning_rate=1e-2, nb_epochs=100):
        # Filter info:
        self.filter_size = filter_size
        self.filter_stride = filter_stride
        self.filter_count = filter_count
        # Pool info:
        self.pool_size = pool_size
        self.pool_stride = pool_stride

        # Convolution step:
        self.filters1 = initialize_filters(filter_size=self.filter_size, filter_count=self.filter_count)
        self.filters2 = initialize_filters(filter_size=self.filter_size, filter_count=self.filter_count)
        self.feature_maps = []
        self.pooled_layers = []
        self.pool_shape = None

        # Fully-connected layers:
        self.w1 = np.random.uniform(-1, 1, fully1_size * fully2_size).reshape(fully1_size, fully2_size)
        self.b1 = np.zeros(16)
        self.w2 = np.random.uniform(-1, 1, fully2_size * output_size).reshape(output_size, fully2_size)
        self.b2 = np.zeros(3)

        # Training hyperparameters:
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs

    def fit(self, x_train, y_train):
        for epoch in range(self.nb_epochs):
            print("Epoch " + str(epoch))
            for i, image in enumerate(x_train):
                # Forward-prop:
                # ----------------
                conv1 = convolve([image], self.filters1, 1)
                conv2 = convolve(conv1, self.filters2, 1)

                pooled = max_pool(conv2, self.pool_size, self.pool_stride)
                self.pool_shape = pooled.shape

                flat = flatten_layer(pooled)
                flat = std_scaler(flat)
                hidden = feed_forward(self.w1, self.b1, flat)
                final_layer = feed_forward(self.w2, self.b2, hidden)
                activation = softmax(final_layer)
                y_pred_ind = np.argmax(activation)
                y_pred = np.zeros(activation.shape)
                y_pred[y_pred_ind] = 1.0

                y_actual = y_train[i]

                # Loss:
                # ----------------
                loss = categorical_cross_entropy(activation, y_train[i])
                # Back-prop:
                # ----------------
                # Dense layers:
                dLda = activation - y_actual

                # d/dw(w.T@x + b) = x = hidden
                dadw2 = hidden
                dadb2 = np.ones(len(final_layer))

                dLdw2 = np.array([dLda]).T * dadw2

                # d/dx(w.T@x + b) = w = self.w2
                dadx2 = self.w2

                dLdx2 = np.inner(dLda, dadx2.T)
                dadw1 = flat
                dadb1 = np.ones(len(hidden))
                dLdw1 = np.array([dLdx2]).T * dadw1

                dx2dx1 = self.w1
                dLdx1 = np.dot(dLdx2, dx2dx1)

                # Updating weights
                self.w2 -= self.learning_rate * dLdw2
                self.b2 -= self.learning_rate * dLda * dadb2

                self.w1 -= self.learning_rate * dLdw1
                self.b1 -= self.learning_rate * dadb1 * dLdx2
                # Convolutional layers:
                # ---------------------
                # Pooling backprop:
                # This is basically a gradient "switch". Putting the gradients to the points
                # that contained the maximum because they had 100% contribution whereas the rest had 0%
                repooled = return_dimensions(flat, self.pool_shape)
                repooled_gradients = return_dimensions(dLdx1, self.pool_shape)
                gradient_grid2 = np.zeros(conv2.shape)
                gradient_grid2 = backward_maxpool(gradient_grid2, repooled, self.pool_stride, conv2, repooled_gradients)

                # Conv layer backprop:

                dLdF2_1, dLdF2_2 = backward_convolution_wrt_weights(gradient_grid2, conv1, self.filter_count,
                                                                    self.filter_size, self.filter_stride)

                self.filters2[0] -= self.learning_rate * dLdF2_1
                self.filters2[1] -= self.learning_rate * dLdF2_2

                dLdI2 = backward_convolution_wrt_inputs(gradient_grid2, self.filters2, conv1.shape, self.filter_stride)

                dLdF1_1, dLdF1_2 = backward_convolution_wrt_weights(dLdI2, [image], self.filter_count, self.filter_size,
                                                                    self.filter_stride)

                self.filters1[0] -= self.learning_rate * dLdF1_1
                self.filters1[1] -= self.learning_rate * dLdF1_2

    def predict(self, X, probabilities=False):
        # Convolution
        decisions = []
        for sample in X:
            conv1 = convolve([sample], self.filters1, self.filter_stride)
            conv2 = convolve(conv1, self.filters2, self.filter_stride)
            # Max Pooling
            pooled = max_pool(conv2, self.pool_size, self.pool_stride)
            # Flattening
            flat = flatten_layer(pooled)
            # Regular NN
            flat = std_scaler(flat)
            hidden = feed_forward(self.w1, self.b1, flat)
            final_layer = feed_forward(self.w2, self.b2, hidden)
            activation = softmax(final_layer)
            if probabilities is True:
                decisions.append(activation)
            else:
                decision = np.zeros(activation.shape)
                index = np.argmax(activation)
                decision[index] = 1.0
                decisions.append(decision)
        return np.asarray(decisions)


smile1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]])

smile2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]])

smile3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0, 1, 1, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]])

frown1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 0, 1, 1, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]])

frown2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 0, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]])

frown3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]])

ok1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 1, 1, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]])

ok2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 1, 1, 1, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]])

ok3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 1, 1, 1, 1, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]])




# Generating dataset:

data_train = []
label_train = []

data_train.append(smile1)
data_train.append(frown1)
data_train.append(ok1)
label_train.append(np.array([1., 0., 0.]))
label_train.append(np.array([0., 0., 1.]))
label_train.append(np.array([0., 1., 0.]))
data_train.append(smile2)
data_train.append(frown2)
data_train.append(ok2)
label_train.append(np.array([1., 0., 0.]))
label_train.append(np.array([0., 0., 1.]))
label_train.append(np.array([0., 1., 0.]))
data_train.append(smile3)
data_train.append(frown3)
data_train.append(ok3)
label_train.append(np.array([1., 0., 0.]))
label_train.append(np.array([0., 0., 1.]))
label_train.append(np.array([0., 1., 0.]))

# Fitting:
clf = CNN(nb_epochs=10)

clf.fit(data_train, label_train)

face1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 1, 0],
                 [0, 1, 0, 0, 0, 0, 1, 0],
                 [0, 0, 1, 0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]])

face2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]])

face3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 1, 1, 0, 0, 0],
                 [0, 0, 1, 0, 1, 1, 0, 0],
                 [0, 1, 0, 0, 0, 1, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]])

predict_mood(face1, clf)
predict_mood(face2, clf)
predict_mood(face3, clf)

# Check out if filters are interpretable 
print(clf.filters1)
print(clf.filters2)


