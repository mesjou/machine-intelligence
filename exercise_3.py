import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def tanh_derivative(x):
    return 1 - np.square(np.tanh(x))


def mse(prediction, target):
    return 0.5 * np.square(target - prediction)


def mse_derivative(prediction, target):
    return -target + prediction


class MLP:
    def __init__(self):

        # hyperparameters
        self.learning_rate = 0.5

        # init input layer weights
        self.b_input = np.random.uniform(-0.5, 0.5, (3, 1))
        self.w_input = np.random.normal(-0.5, 0.5, (1, 3))

        # init hidden layer weights
        self.b_layer = np.random.uniform(-0.5, 0.5, (1, 1))
        self.w_layer = np.random.normal(-0.5, 0.5, (3, 1))

        # init gradients
        self.init_gradients()

    def init_gradients(self):
        # init input layer gradients
        self.b_input_grad = np.zeros((3, 1))
        self.w_input_grad = np.zeros((1, 3))

        # init hidden layer gradients
        self.b_layer_grad = np.zeros((1, 1))
        self.w_layer_grad = np.zeros((3, 1))

    def forward(self, x):
        # get input of hidden layer
        x = np.reshape(x, (1, 1))
        h_layer = np.dot(self.w_input.transpose(), x) + self.b_input

        # get activity of hidden layer
        s_layer = np.tanh(h_layer)

        # get input/activity of output layer
        y = np.dot(self.w_layer.transpose(), s_layer) + self.b_layer

        return h_layer, s_layer, y

    def predict(self, x: np.array) -> np.array:
        """Return only the prediction of the forward pass."""
        return self.forward(x)[-1]

    def backpropagate(self, x, y_hat, y, h_layer):
        # get derivatives
        delta_output = 1.0
        delta_w_layer = np.multiply(tanh_derivative(h_layer), self.w_layer).transpose()  # delta_output = 1.0
        delta_b_layer = self.b_layer

        # calculate gradients
        derror = mse_derivative(y_hat, y)[0][0]
        self.b_input_grad += derror * delta_b_layer  # x at bias = 1.0
        self.w_input_grad += derror * delta_w_layer * x  # identity activation function in input layer f(h) = x
        self.b_layer_grad += derror * delta_output * 1.0  # activation von bias is f(h0) = 1
        self.w_layer_grad += derror * delta_output * np.tanh(h_layer)

    def update(self, batch_size):
        self.b_input -= self.learning_rate * self.b_input_grad / batch_size
        self.w_input -= self.learning_rate * self.w_input_grad / batch_size
        self.b_layer -= self.learning_rate * self.b_layer_grad / batch_size
        self.w_layer -= self.learning_rate * self.w_layer_grad / batch_size

    def train(self, xx, yy):
        error_logging = []
        Done = False
        t = 0
        last_periods_error = 10
        while not Done:
            self.init_gradients()
            for x, y in zip(xx, yy):
                h_layer, s_layer, y_hat = self.forward(x)
                self.backpropagate(x, y_hat, y, h_layer)
            self.update(len(yy))
            t += 1

            # check convergence
            error = []
            for x, y in zip(xx, yy):
                error.append(mse(self.predict(x), y))
            error = np.mean(error)
            error_logging.append(error)
            if np.abs(error - last_periods_error) / error < 1e-5 or t == 3000:
                Done = True
            last_periods_error = np.mean(error)
        return error_logging


if __name__ == "__main__":
    data = pd.read_csv("data/RegressionData.txt", sep=" ", header=None)
    data.columns = ["x", "y"]
    X = data.x.values
    Y = data.y.values
    print(data)

    mlp = MLP()
    errors = mlp.train(X, Y)
    print("")
    pass

    # plot error
    plt.plot(errors, label="error")
    plt.legend()
    plt.show()

    # plot hidden layer output
    plt.scatter(X, Y, label="data")
    for i in range(3):
        plt.plot(
            np.linspace(0, 1, 50), [mlp.forward(x)[1][i] for x in np.linspace(0, 1, 50)], label="neuron_{}".format(i)
        )
    plt.legend()
    plt.show()

    # plot prediction
    plt.scatter(X, Y, label="data")
    plt.plot(np.linspace(0, 1, 50), [mlp.predict(x)[0][0] for x in np.linspace(0, 1, 50)], label="prediction", color="orange")
    plt.legend()
    plt.show()
