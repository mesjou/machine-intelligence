from keras import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras import optimizers
from keras.losses import BinaryCrossentropy
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # generate data
    x = np.random.randint(low=0, high=9, size=(10000, 30))
    y = np.where(x.sum(axis=1) >= 100.0, 1.0, 0.0)
    y = np.reshape(y, (10000, 1))
    x_train, x_test = np.split(x, [8000], axis=0)
    y_train, y_test = np.split(y, [8000], axis=0)

    # specify model
    model = Sequential()
    model.add(LSTM(200, input_shape=(30, 1), return_sequences=False))
    model.add(Dense(1, activation="linear"))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()

    units = 200
    W = model.layers[0].get_weights()[0]
    U = model.layers[0].get_weights()[1]
    b = model.layers[0].get_weights()[2]

    W_i = W[:, :units]
    W_f = W[:, units : units * 2]
    W_c = W[:, units * 2 : units * 3]
    W_o = W[:, units * 3 :]

    U_i = U[:, :units]
    U_f = U[:, units : units * 2]
    U_c = U[:, units * 2 : units * 3]
    U_o = U[:, units * 3 :]

    b_i = b[:units]
    b_f = b[units : units * 2]
    b_c = b[units * 2 : units * 3]
    b_o = b[units * 3 :]

    # train model
    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        ),
        metrics=["accuracy"],
    )
    history = model.fit(x_train, y_train, epochs=60, batch_size=50, verbose=0)
    plt.plot(history.history["accuracy"])
    plt.title("Accuracy")
    plt.show()

    plt.plot(history.history["loss"])
    plt.title("Loss")
    plt.show()

    scores = model.evaluate(x_test, y_test, verbose=1, batch_size=1)
    print("Accurracy: {}".format(scores[1]))
    from sklearn.metrics import confusion_matrix

    y_pred = model.predict(x_test)
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
