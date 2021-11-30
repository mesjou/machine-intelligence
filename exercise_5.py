import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold


def sphere(W, X):
    """Sphere the data so that data has mean zero, unit variance and is uncorrelated.

    :param W: matrix to be sphered
    :param X: matrix from which Lambda, Eig etc are calculated
    :return: sphered data matrix
    """

    # center data
    centered_W = W - np.mean(X, axis=1, keepdims=True)

    # decorrelate and adjust variance
    covariance = np.cov(centered_W, rowvar=True, bias=True)
    eigenvalues, eigenvector = np.linalg.eig(covariance)

    return np.diag(np.power(eigenvalues, -0.5)).dot(eigenvector.transpose()).dot(centered_W)


def scatter_plot(X, Y, title, clip=True):
    plt.scatter(X[0, :], X[1, :], c=Y)
    plt.ylabel("X1")
    plt.xlabel("X2")
    if clip is True:
        plt.clim(0, 50)
    plt.colorbar()
    plt.title(title)
    plt.show()


def monomials(X, K):
    phi = []
    for k in range(K + 1):
        for l in range(k + 1):
            m = k - l
            phi.append(np.power(X[0, :], m) * np.power(X[1, :], l))
    return np.array(phi)


if __name__ == "__main__":
    train = pd.read_csv("data/TrainingRidge.csv")
    val = pd.read_csv("data/ValidationRidge.csv")

    x_train = np.array([train.x1.values, train.x2])
    y_train = np.array(train.obs.values)

    x_validation = np.array([val.x1.values, val.x2])
    y_validation = np.array(val.dens.values)

    # sphere training data
    x_train_sphered = sphere(x_train, x_train)
    print('Mean:\n', np.mean(x_train_sphered, axis=1))
    print('Variance: \n', np.var(x_train_sphered, axis=1))
    print('Covariance Matrix: \n', np.cov(x_train_sphered, rowvar=True, bias=True))
    scatter_plot(x_train, y_train, "Training set unsphered")
    scatter_plot(x_train_sphered, y_train, "Training set sphered")

    # sphere validation data
    x_validation_sphered = sphere(x_validation, x_train)
    print('Mean:\n', np.mean(x_validation_sphered, axis=1))
    print('Variance: \n', np.var(x_validation_sphered, axis=1))
    print('Covariance Matrix: \n', np.cov(x_validation_sphered, rowvar=True, bias=True))
    scatter_plot(x_validation, y_validation, "Validation set unsphered")
    scatter_plot(x_validation_sphered, y_validation, "Validation set sphered")

    # b
    phi_train = monomials(x_train_sphered, 9)
    phi_validation = monomials(x_validation_sphered, 9)

    # i
    for order in range(10):
        pass
        scatter_plot(x_validation_sphered, phi_validation[order, :], "Monomial of order: {}".format(order + 1))

    # ii)
    w_star = np.linalg.inv(phi_train.dot(phi_train.transpose())).dot(phi_train).dot(y_train.transpose())
    y_predicted = w_star.transpose().dot(phi_train)
    scatter_plot(x_train_sphered, y_predicted, "Predicted y with monomial of order 9")
    scatter_plot(x_validation_sphered, w_star.transpose().dot(phi_validation), "Predicted y validation set with monomial of order 9")

    # c

    # i)
    row = 0
    K = 10
    Z = np.arange(-4.0, 4.0, 0.1)
    mse = np.zeros(shape=(K, len(Z)))
    kf = KFold(n_splits=K, shuffle=True)
    for train_index, test_index in kf.split(phi_train.transpose(), y_train.transpose()):
        phi_train_cv, phi_test_cv = phi_train[:, train_index], phi_train[:, test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

        # size of Phi needed for identity matrix
        m = phi_train_cv.shape[0]
        for col, z in enumerate(Z):

            # train with regularization
            lambd = 10 ** z
            w_reg = np.linalg.inv(phi_train_cv.dot(phi_train_cv.transpose()) + lambd * np.identity(m)).dot(phi_train_cv).dot(y_train_cv.transpose())

            # get mse
            y_predicted_cv = w_reg.transpose().dot(phi_test_cv)
            mse[row, col] = np.mean((y_test_cv - y_predicted_cv) ** 2)

        row += 1

    # plot results
    mean = np.mean(mse, axis=0)
    std = np.std(mse, axis=0)
    plt.errorbar(Z, mean, std)
    plt.show()

    # ii
    print("Best lambda: ", Z[np.argmin(np.mean(mse, axis=0))], " with MSE: ", np.min(np.mean(mse, axis=0)))
    lambd_star = 10 ** Z[np.argmin(np.mean(mse, axis=0))]

    # iii
    w_star_reg = np.linalg.inv(phi_train.dot(phi_train.transpose()) + lambd_star * np.identity(n=phi_train.shape[0])).dot(phi_train).dot(
        y_train.transpose())
    y_predicted_reg = w_star_reg.transpose().dot(phi_validation)
    mse = (y_validation - y_predicted_reg) ** 2.0
    print("MSE on evaluation set: ", np.mean(mse))
    scatter_plot(x_train, y_train, "Training Set True Y", clip=True)
    scatter_plot(x_train, w_star_reg.transpose().dot(phi_train), "Training Set Predicted Y", clip=True)
    scatter_plot(x_validation, y_validation, "Validation Set True Y", clip=True)
    scatter_plot(x_validation, y_predicted_reg, "Validation Set Predicted Y", clip=True)

    plt.plot(y_predicted_reg, label="predicted y")
    plt.plot(y_validation, label="true y")
    plt.legend()
    plt.title("Correct Cross Validation")
    plt.show()

    # d
    row = 0
    K = 10
    Z = np.arange(-4.0, 4.0, 0.1)
    mse = np.zeros(shape=(K, len(Z)))
    kf = KFold(n_splits=K, shuffle=True)
    for train_index, test_index in kf.split(phi_validation.transpose(), y_validation.transpose()):
        phi_train_cv, phi_test_cv = phi_validation[:, train_index], phi_validation[:, test_index]
        y_train_cv, y_test_cv = y_validation[train_index], y_validation[test_index]

        # size of Phi needed for identity matrix
        m = phi_train_cv.shape[0]
        for col, z in enumerate(Z):

            # train with regularization
            lambd = 10 ** z
            w_reg = np.linalg.inv(phi_train_cv.dot(phi_train_cv.transpose()) + lambd * np.identity(m)).dot(phi_train_cv).dot(y_train_cv.transpose())

            # get mse
            y_predicted_cv = w_reg.transpose().dot(phi_test_cv)
            mse[row, col] = np.mean((y_test_cv - y_predicted_cv) ** 2)

        row += 1

    # plot results
    mean = np.mean(mse, axis=0)
    std = np.std(mse, axis=0)
    plt.errorbar(Z, mean, std)
    plt.show()

    # ii
    print("Best lambda with validation set: ", Z[np.argmin(np.mean(mse, axis=0))], " with MSE: ", np.min(np.mean(mse, axis=0)))
    lambd_star = 10 ** Z[np.argmin(np.mean(mse, axis=0))]

    # iii
    w_star_reg = np.linalg.inv(phi_validation.dot(phi_validation.transpose()) + lambd_star * np.identity(n=phi_validation.shape[0])).dot(phi_validation).dot(
        y_validation.transpose())
    y_predicted_reg = w_star_reg.transpose().dot(phi_validation)
    mse = (y_validation - y_predicted_reg) ** 2.0
    print("MSE on evaluation set with validation as training: ", np.mean(mse))

    # e
    scatter_plot(x_train, y_train, "Training Set True Y (validation as training)", clip=True)
    scatter_plot(x_train, w_star_reg.transpose().dot(phi_train), "Training Set Predicted Y (validation as training)", clip=True)
    scatter_plot(x_validation, y_validation, "Validation Set True Y (validation as training)", clip=True)
    scatter_plot(x_validation, y_predicted_reg, "Validation Set Predicted Y (validation as training)", clip=True)

    plt.plot(y_predicted_reg, label="predicted y")
    plt.plot(y_validation, label="true y")
    plt.legend()
    plt.title("Calidation as training Cross Validation")
    plt.show()
