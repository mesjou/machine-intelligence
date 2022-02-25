from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras import layers
from keras import initializers
from keras import activations
from tensorflow.keras import optimizers
from keras import losses
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


class MetricsCallback(Callback):
    def __init__(self, x_train, x_evaluate, y_train, y_evaluate):
        # save data
        self.x_train = x_train
        self.x_evaluate = x_evaluate
        self.y_train = y_train
        self.y_evaluate = y_evaluate

        # save metrices
        self.train_error = []
        self.evaluation_error = []
        self.accuracy = tf.keras.metrics.CategoricalAccuracy()

    def on_epoch_end(self, epoch, logs=None):
        """Evaluate test and evaluation accuracy at each epoch."""
        y_pred = self.model.predict(self.x_train)
        self.accuracy.reset_state()
        self.accuracy.update_state(self.y_train, y_pred)
        self.train_error.append(self.accuracy.result().numpy())

        y_pred = self.model.predict(self.x_evaluate)
        self.accuracy.reset_state()
        self.accuracy.update_state(self.y_evaluate, y_pred)
        self.evaluation_error.append(self.accuracy.result().numpy())


# linear model
linear_model = keras.Sequential()
linear_model.add(
    layers.Dense(
        units=10,
        kernel_initializer=initializers.Zeros(),
        bias_initializer=initializers.Zeros(),
        activation=activations.softmax,
    )
)
linear_model.compile(
    optimizer=optimizers.SGD(learning_rate=0.5),
    loss=losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)


# Multi-Layer Perceptton with Non-Linearities
mlp_model = keras.Sequential()
mlp_model.add(
    layers.Dense(
        units=1500,
        kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
        bias_initializer=initializers.Constant(0.1),
        activation=activations.relu,
    )
)
mlp_model.add(
    layers.Dense(
        units=1500,
        kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
        bias_initializer=initializers.Constant(0.1),
        activation=activations.relu,
    )
)
mlp_model.add(
    layers.Dense(
        units=1500,
        kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
        bias_initializer=initializers.Constant(0.1),
        activation=activations.relu,
    )
)
mlp_model.add(
    layers.Dense(
        units=10,
        kernel_initializer=initializers.TruncatedNormal(stddev=0.01),
        bias_initializer=initializers.Constant(0.1),
        activation=activations.softmax,
    )
)
mlp_model.compile(
    optimizer=optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
    ),
    loss=losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # one-hot-encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # reshape data so that we have an input vector of length 28x28=784
    feature_length = 784
    x_train = x_train.reshape(x_train.shape[0], feature_length)
    x_test = x_test.reshape(x_test.shape[0], feature_length)

    # 1. Linear model
    # 60.000 datapoints, batch size=100-> we need 600 iterations to complete one epoch
    # -> we need around 17 epochs to complete 10.000 iterations
    acurracy_history = MetricsCallback(
        x_train=x_train, x_evaluate=x_test, y_train=y_train, y_evaluate=y_test
    )
    linear_history = linear_model.fit(
        x=x_train, y=y_train, batch_size=100, epochs=17, callbacks=[acurracy_history]
    )
    linear_model.evaluate(x=x_test, y=y_test)
    plt.plot(acurracy_history.evaluation_error, label="Evaluation Accuracy")
    plt.plot(acurracy_history.train_error, label="Training Accuracy")
    plt.legend()
    plt.title("Linear Model")
    plt.show()

    # 2. MLP
    # 60.000 datapoints, batch size=100-> we need 600 iterations to complete one epoch
    # -> we need around 33 epochs to complete 20.000 iterations
    acurracy_history = MetricsCallback(
        x_train=x_train, x_evaluate=x_test, y_train=y_train, y_evaluate=y_test
    )
    mlp_history = mlp_model.fit(
        x=x_train, y=y_train, batch_size=100, epochs=33, callbacks=[acurracy_history]
    )
    mlp_model.evaluate(x=x_test, y=y_test)
    plt.plot(acurracy_history.evaluation_error, label="Training Accuracy")
    plt.plot(acurracy_history.train_error, label="Evaluation Accuracy")
    plt.legend()
    plt.title("MLP")
    plt.show()
