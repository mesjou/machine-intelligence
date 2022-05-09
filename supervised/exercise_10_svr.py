# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 14:11:56 2022

@author: User
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import NuSVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


#%% Load Data, fit model, test on validation set
directory = os.getcwd()
train = pd.read_csv(directory + "/data/TrainingRidge.csv")
val = pd.read_csv(directory + "/data/ValidationRidge.csv")

x_train = np.array([train.x1.values, train.x2]).transpose()
y_train = np.array(train.obs.values)

x_validation = np.array([val.x1.values, val.x2]).transpose()
y_validation = np.array(val.dens.values)

svr_model = NuSVR()
svr_model.fit(x_train, y_train)

# predict on validation and calculate error
y_pred = svr_model.predict(x_validation)
err = (y_pred - y_validation) ** 2 / len(y_pred)
print("Train MSE: ", mean_squared_error(svr_model.predict(x_train), y_train))
print("Test MSE: ", mean_squared_error(y_pred, y_validation))

#%% plots

# define axises, reshape y_pred and err for contour plot (must be 2D)
x1_axis, x2_axis = np.meshgrid(np.arange(140, 212, 2), np.arange(40, 122, 2))
y_pred = y_pred.reshape(x1_axis.shape)
err = err.reshape(x1_axis.shape)

# plot predicted output on validatoin set
fig = plt.subplot()
fig.contourf(x1_axis, x2_axis, y_pred)
fig.scatter(x_train[:, 0], x_train[:, 1], marker="s", c="r")
fig.set_xlabel("X1")
fig.set_ylabel("X2")
plt.show()
# fig.colorbar() #not working?

# plot MSE
plt.contourf(x1_axis, x2_axis, err)
plt.xlabel("X1")
plt.ylabel("X2")
plt.colorbar()

#%% Grid Search

nu = 0.5
Cs = [2 ** x for x in np.arange(-2, 12, 2, dtype=float)]
gammas = [2 ** x for x in np.arange(-12, 0, 2, dtype=float)]

mse = []
# for each value of C and gamma fit the model and do cross-validation on training set
for C in Cs:
    for gamma in gammas:
        svr_model = NuSVR(C=C, gamma=gamma, nu=nu)
        scores = cross_val_score(
            svr_model, x_train, y_train, cv=10, scoring="neg_mean_squared_error"
        )
        mean_mse = np.mean(scores)
        mse = np.append(mse, mean_mse)

# find parameters C and gamma that minimize MSE (i.e maximize negative mse)
mse = mse.reshape((len(Cs), len(gammas)))
min_param = np.unravel_index(np.argmax(mse), np.array(mse).shape)
print("Min paramters at index: ", min_param, "with mean MSE: ", -1 * mse[min_param])

C_min = Cs[min_param[0]]
gamma_min = gammas[min_param[1]]
print("TRUE VALUES C: ", C_min, "Gamma: ", gamma_min)
print("LOG VALUES  C: ", np.log(C_min), "Gamma: ", np.log(gamma_min))

svr_model = NuSVR(C=C_min, gamma=gamma_min, nu=nu)
svr_model.fit(x_train, y_train)

y_pred = svr_model.predict(x_validation)
err = (y_pred - y_validation) ** 2 / len(y_pred)
print("Train MSE: ", mean_squared_error(svr_model.predict(x_train), y_train))
print("Test MSE: ", mean_squared_error(y_pred, y_validation))

#%% plot

y_pred = y_pred.reshape(x1_axis.shape)
err = err.reshape(x1_axis.shape)

fig = plt.subplot()
fig.contourf(x1_axis, x2_axis, y_pred)
fig.set_xlabel("X1")
fig.set_ylabel("X2")
plt.show()
# fig.colorbar() #not working?

# plot MSE
plt.contourf(x1_axis, x2_axis, err)
plt.xlabel("X1")
plt.ylabel("X2")
plt.colorbar()
