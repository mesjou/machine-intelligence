import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def pca(data):

    # centering
    data_centered = data - np.mean(data, axis=0)
    plt.scatter(data_centered[:, 0], data_centered[:, 1])
    plt.show()

    # eigenvalue decomposition of covariance matrix
    cov = np.cov(data.transpose())

    # eigenvalues lambda and right eigenvectors eig
    lamb, eig = np.linalg.eig(cov)

    # sort according to eigenvalues (highest first)
    idx = np.argsort(lamb)
    eig_sorted = eig[:, idx]

    # check (see slides p. 15)
    a_1 = np.transpose(eig[:, 0]) @ data_centered[0, :]
    a_2 = np.transpose(eig[:, 1]) @ data_centered[0, :]
    x_recovered = a_1 * eig[:, 0] + a_2 * eig[:, 1]
    assert np.array(
        np.round(data_centered[0, :] - x_recovered, 4) == 0
    ).all(), "Value should be 0 but is {}".format(data_centered[0, :] - x_recovered)

    return eig_sorted, data_centered


def remove_outlier(data, n_std=3):

    # calculate summary statistics
    data_mean, data_std = np.mean(data, axis=0), np.std(data, axis=0)

    # identify outliers
    cut_off = data_std * n_std
    lower, upper = data_mean - cut_off, data_mean + cut_off

    # identify outliers
    clean_data = np.array(
        [x for x in data if np.array(x >= lower).all() and np.array(x <= upper).all()]
    )
    print(f"Non-outlier observations: {len(clean_data)} from {len(data)}")

    return clean_data


def scree_plot(pca, title):
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, "o-", linewidth=2, color="blue")
    plt.title("Scree Plot " + title)
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.show()


if __name__ == "__main__":

    # exercise 1

    # load data
    data = np.loadtxt(os.getcwd() + "/data/pca2.csv", delimiter=",", skiprows=1)
    eig, data_centered = pca(data)

    # scatter plot along PC1 and PC2 axes
    A_1 = data_centered @ np.transpose(eig[:, 0])
    A_2 = data_centered @ np.transpose(eig[:, 1])
    plt.scatter(A_1, A_2)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("With Outlier")
    plt.show()

    # load data
    index = [16, 156]
    data_cleaned = np.delete(data, index, axis=0)
    eig, data_centered = pca(data_cleaned)

    # scatter plot along PC1 and PC2 axes
    A_1 = data_centered @ np.transpose(eig[:, 0])
    A_2 = data_centered @ np.transpose(eig[:, 1])
    plt.scatter(A_1, A_2)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Without Outlier")
    plt.show()

    # exercise 2

    # load data
    data = np.loadtxt(os.getcwd() + "/data/pca4.csv", delimiter=",", skiprows=1)
    data_centered = data - np.mean(data, axis=0)

    # plot scatter matrix
    data_df = pd.DataFrame(data_centered, columns=["x1", "x2", "x3", "x4"])
    pd.plotting.scatter_matrix(data_df, alpha=0.2, figsize=(6, 6), diagonal="kde")
    plt.show()

    # remove outlier
    data_cleaned = remove_outlier(data)
    data_cleaned = data_cleaned - np.mean(data_cleaned, axis=0)
    data_df = pd.DataFrame(data_cleaned, columns=["x1", "x2", "x3", "x4"])
    pd.plotting.scatter_matrix(data_df, alpha=0.2, figsize=(6, 6), diagonal="kde")
    plt.show()

    # apply PCA to cleaned data and scree plot
    pca = PCA(n_components=4)
    pca.fit(data_cleaned)
    scree_plot(pca, "Cleaned data")

    # whitening

    # eigenvalue decomposition of covariance matrix
    # eigenvalues lambda and right eigenvectors eig
    cov = np.cov(data_cleaned.transpose())
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    whitened_data = (
        np.diag(1.0 / np.sqrt(eigenvalues)).dot(eigenvectors.T).dot(data_cleaned.T)
    )

    # covariance plot
    plt.imshow(cov, cmap="viridis")
    plt.colorbar()
    plt.title("Cov of centered data")
    plt.show()

    # covariance of projected data
    plt.imshow(np.cov(eigenvectors.T.dot(data_cleaned.T)), cmap="viridis")
    plt.colorbar()
    plt.title("Cov of projection PC")
    plt.show()

    # covariance of whitened data
    plt.imshow(np.cov(whitened_data), cmap="viridis")
    plt.colorbar()
    plt.title("Cov of whitened data")
    plt.show()
