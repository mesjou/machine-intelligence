import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
import copy


class Neuron:
    def __init__(self):
        self.weights = np.array([-0.45, 0.2]).transpose()
        self.learning_rate = 0.1

    def whoami(self):
        return type(self).__name__

    def get_weights(self):
        return self.weights[0], self.weights[1]

    def gradient(self, x, y):
        hessian = x.dot(x.transpose())
        return hessian.dot(self.weights) - x.dot(y.transpose())

    @abstractmethod
    def train(self, x, y):
        pass


class GradientNeuron(Neuron):
    def train(self, x, y):
        self.weights -= self.learning_rate * self.gradient(x, y)


class LineSearchNeuron(Neuron):
    def guide_learning_rate(self, x, y):
        g = self.gradient(x, y)
        hessian = x.dot(x.transpose())
        denominator = g.transpose().dot(hessian).dot(g)
        nominator = g.transpose().dot(g)
        if denominator == 0.0:
            self.learning_rate = 0.0
        else:
            self.learning_rate = nominator / denominator

    def train(self, x, y):
        self.guide_learning_rate(x, y)
        self.weights -= self.learning_rate * self.gradient(x, y)


class ConjugateNeuron(Neuron):
    def __init__(self):
        super().__init__()
        self.direction = False
        self.old_gradient = False

    def guide_learning_rate(self, x, y):
        hessian = x.dot(x.transpose())
        denominator = self.direction.transpose().dot(hessian).dot(self.direction)
        nominator = self.direction.transpose().dot(self.old_gradient)
        if denominator == 0.0:
            self.learning_rate = 0.0
        else:
            self.learning_rate = - nominator / denominator

    def train(self, x, y):
        if self.direction is False:
            gradient = self.gradient(x, y)
            self.direction = - gradient
            self.old_gradient = gradient

        # update weights
        self.guide_learning_rate(x, y)
        self.weights += self.learning_rate * self.direction

        # update direction
        new_gradient = self.gradient(x, y)
        denominator = self.old_gradient.transpose().dot(self.old_gradient)
        nominator = new_gradient.transpose().dot(new_gradient)
        if denominator == 0.0:
            beta = 0.0
        else:
            beta = - nominator / denominator
        self.direction = new_gradient + beta * self.direction
        self.old_gradient = new_gradient


def plot(neuron: Neuron, T=30):
    w_input = []
    w_layer = []
    for t in range(T):
        w_input.append(neuron.get_weights()[0])
        w_layer.append(neuron.get_weights()[1])
        neuron.train(x, y)

    plt.plot(w_input, label="w0")
    plt.plot(w_layer, label="w1")
    plt.xlabel("Period")
    plt.legend()
    plt.title(neuron.whoami())
    plt.show()

    plt.scatter(w_input, w_layer, c=np.arange(T))
    plt.ylabel("Weight 0")
    plt.xlabel("Weight 1")
    plt.colorbar()
    plt.title(neuron.whoami())
    plt.show()


if __name__ == "__main__":
    x = np.array([[1, 1, 1], [-1, 0.3, 2]])
    y = np.array([-0.1, 0.5, 0.5])
    plot(GradientNeuron())
    plot(LineSearchNeuron())
    plot(ConjugateNeuron())
