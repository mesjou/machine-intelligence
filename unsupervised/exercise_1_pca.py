# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
import pandas as pd

# exercise 1

# load data
data = np.loadtxt(os.getcwd() + "/data/pca-data-2d.dat")

# centering
data_centered = data - np.mean(data, axis=0)

# a)
plt.scatter(data_centered[:, 0], data_centered[:, 1])
plt.show()

# b) pca
pca = PCA(n_components=2)
pca.fit(data_centered)
print(pca.components_)

# check (see slides p. 15)
pca_components = pca.components_
a_1 = np.transpose(pca_components[:, 0]) @ data_centered[0, :]
a_2 = np.transpose(pca_components[:, 1]) @ data_centered[0, :]
x_recovered = a_1 * pca_components[:, 0] + a_2 * pca_components[:, 1]
print(data_centered[0, :] - x_recovered)

A_1 = data_centered @ np.transpose(pca_components[:, 0])
A_2 = data_centered @ np.transpose(pca_components[:, 1])

plt.scatter(A_1, A_2)
plt.show()

# alternative
data_pca = pca.transform(data_centered)
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.show()

# c)
# i) first PC: X^tilde=a1*e1
data_reconstructed_1 = np.array([a_1 * pca_components[:, 0] for a_1 in A_1])
plt.scatter(data_reconstructed_1[:, 0], data_reconstructed_1[:, 1])
plt.show()

# ii) second PC
data_reconstructed_2 = np.array([a_2 * pca_components[:, 0] for a_2 in A_2])
plt.scatter(data_reconstructed_2[:, 0], data_reconstructed_2[:, 1])
plt.show()  # why is this a straight line??


# Exercise 2

# a)
# load data and center
data = np.loadtxt(os.getcwd() + "/data/pca-data-3d.txt", delimiter=",", skiprows=1)
data_centered = data - np.mean(data, axis=0)
data_df = pd.DataFrame(data_centered, columns=["x", "y", "z"])

# plot scatter matrix
pd.plotting.scatter_matrix(data_df, alpha=0.2, figsize=(6, 6), diagonal="kde")
plt.show()

# b)

# apply PCA
pca = PCA(n_components=3)
transformed_data = pca.fit_transform(data_centered)

# plot scatter matrix on PCs
data_df = pd.DataFrame(transformed_data, columns=["PC1", "PC2", "PC3"])
pd.plotting.scatter_matrix(data_df, alpha=0.2, figsize=(6, 6), diagonal="kde")
plt.show()

# c
def plot_3d(x, y, z, title):
    sns.set(style="darkgrid")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.scatter(x, y, z)

    plt.title(title)
    plt.show()


for n_pca in [1, 2, 3]:
    pca = PCA(n_components=n_pca)
    transformed_data = pca.fit_transform(data_centered)
    reconstructed_data = pca.inverse_transform(transformed_data)
    plot_3d(
        reconstructed_data[:, 0],
        reconstructed_data[:, 1],
        reconstructed_data[:, 2],
        "Used PCAs: " + str(n_pca),
    )


# Exercise 3

# array-like of shape (n_samples, n_features)
# 100 data points with 20 features
data = np.loadtxt(
    os.getcwd() + "/data/expDat.txt", delimiter=",", skiprows=1, usecols=range(1, 21)
)

# a) pca
pca = PCA(n_components=20)
pca.fit(data)
print(pca.components_)
pca_components = pca.components_

# b)i)scatter plot
A_1 = data @ np.transpose(pca_components[:, 0])
A_2 = data @ np.transpose(pca_components[:, 1])

fig, ax = plt.subplots()
scatter = ax.scatter(A_1, A_2, c=range(100))
legend1 = ax.legend(*scatter.legend_elements(num=10), loc="upper left", title="time")
ax.add_artist(legend1)

# i)line plot
plt.plot(range(100), A_1, label="PC1")
plt.plot(range(100), A_2, label="PC2")
plt.ylabel("Projection onto PCs")
plt.xlabel("time")
plt.legend()
plt.show()

# c)shuffle data

idx = list(range(20))
random.shuffle(idx)
data_shuffled = data[:, idx]

# covariance matrix
cov_data = np.cov(data)
cov_data_shuffled = np.cov(data_shuffled)
print(cov_data)
print(cov_data_shuffled)
# they are the same
sns.heatmap(cov_data)
sns.heatmap(cov_data_shuffled)

# scree plot -> the same for both matrices
eig = np.linalg.eigvals(cov_data)
eig_shuffled = np.linalg.eigvals(cov_data_shuffled)
plt.plot(range(100), eig, label="original data")
plt.plot(range(100), eig_shuffled, label="shuffled data")
plt.ylabel("eigenvalues")
plt.xlabel("time")
plt.legend()
plt.show()
# take away: shuffleing over features does not change anything -> good!

# d)shuffleing over rows (i.e time points) also does not change the covariance matrix


# Exercise 4

import glob
from matplotlib import image

# a)
# load data and center


def sample_patches(start_letter):

    # get image name
    image_dir = os.getcwd() + "/data/imgpca/"
    image_name = str(start_letter) + "*.jpg"
    files = glob.glob(os.path.join(image_dir, image_name))
    assert len(files) > 0.0, "NO IMAGES FOUND"

    # load image and smaple
    patch_size = (16, 16)
    patches = np.empty((0,) + patch_size)

    for file in files:
        data = image.imread(file)

        # display the array of pixels as an image
        # pyplot.imshow(data)
        # pyplot.show()

        # sample patches
        new_patches = extract_patches_2d(
            image=data, patch_size=(16, 16), max_patches=500
        )
        patches = np.concatenate((patches, new_patches), axis=0)

    print("Patches size: ", patches.shape)

    return patches


building_patches = sample_patches("b")
nature_patches = sample_patches("n")

# b)


def plot_pca_direction(pca):
    fig, axes = plt.subplots(4, 6)
    for i, ax in enumerate(axes.flatten()):
        component = pca.components_[i, :]
        ax.imshow(component.reshape((16, 16)))
    plt.show()


# apply PCA to buildings
pca_buildings = PCA(n_components=24)
pca_buildings.fit(building_patches.reshape(*building_patches.shape[:-2], -1))
plot_pca_direction(pca_buildings)

# apply PCA to nature
pca_nature = PCA(n_components=24)
transformed_data = pca_nature.fit(
    nature_patches.reshape(*nature_patches.shape[:-2], -1)
)
plot_pca_direction(pca_nature)

# c)


def scree_plot(pca, title):
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, "o-", linewidth=2, color="blue")
    plt.title("Scree Plot " + title)
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.show()


scree_plot(pca_buildings, "Buildings")
scree_plot(pca_nature, "Nature")
