# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:15:34 2022

@author: User
"""

import numpy as np
from math import log
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#generate data
#1. training data
sigma = np.array([[0.1,0],[0,0.1]])
mu1 = np.array([0,1])
mu2 = np.array([1,0])
mu3 = np.array([0,0])
mu4 = np.array([1,1])
n = 160
np.random.seed(0)

w = np.random.randint(2, size=n)

X_mu1 = np.random.multivariate_normal(mu1, sigma, n)
X_mu2 = np.random.multivariate_normal(mu2, sigma, n)

X1 = np.empty(shape=(n, 2))
X1[w==1, :] = X_mu1[w==1, :]
X1[w==0, :] = X_mu2[w==0, :]

X_mu3 = np.random.multivariate_normal(mu3, sigma, n)
X_mu4 = np.random.multivariate_normal(mu4, sigma, n)

X2 = np.empty(shape=(n, 2))
X2[w==1, :] = X_mu3[w==1, :]
X2[w==0, :] = X_mu4[w==0, :]

X = np.append(X1,X2, axis=0)
Y = np.append(-1 * np.ones(n), np.ones(n))

# split data into training and hold-out sets
X_train, X_validate, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=101)

#scale data?
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_validate = scaler.transform(X_validate)

#plot
plt.scatter(X_train[:,0], X_train[:,1], c=Y_train)
plt.title("Trainining Set")
plt.show()

plt.scatter(X_validate[:,0], X_validate[:,1], c=Y_test)
plt.title("Validation Set")
plt.show()

svm_model = SVC()
svm_model.fit(X_train, Y_train)

y_pred = svm_model.predict(X_validate)
print("Train accuracy: ", accuracy_score(svm_model.predict(X_train), Y_train))
print("Test accuracy: ", accuracy_score(y_pred, Y_test))

#to visualize classification boundary construct query points
#testing data
x1_axis, x2_axis = np.meshgrid(np.arange(-1,2,0.1),np.arange(-1,2,0.1))
query_points=np.c_[x1_axis.ravel(), x2_axis.ravel()]
n=query_points.shape[0]

#plot classification boundary
Z=svm_model.predict(np.c_[x1_axis.ravel(), x2_axis.ravel()])
Z=Z.reshape(x1_axis.shape)
fig=plt.subplot()
fig.contourf(x1_axis, x2_axis, Z)
fig.scatter(svm_model.support_vectors_[:,0], svm_model.support_vectors_[:,1])

print("Support vectors: ", svm_model.n_support_)

Cs = [2 ** x for x in np.arange(-6, 10, 2, dtype=float)]
gammas = [2 ** x for x in np.arange(-5, 9, 2, dtype=float)]

accuracy=[]
#for each value of C and gamma fit the model and do cross-validation on training set 
for C in Cs:
  for gamma in gammas:
    svm_model=SVC(C=C, gamma=gamma)
    scores = cross_val_score(svm_model, X_train, Y_train, cv=10)
    mean_accuracy=np.mean(scores)
    accuracy=np.append(accuracy, mean_accuracy)
    

#plotting
accuracy=accuracy.reshape((len(Cs), len(gammas)))

#convert axis to log scale
Cs_scaled = [log(x, 2) for x in Cs]
gammas_scaled = [log(x, 2) for x in gammas]

plt.contour(gammas_scaled, Cs_scaled, accuracy)
plt.ylabel("C")
plt.xlabel("Gamma")
plt.colorbar()

#find parameters C and gamma that maximize accuracy
max_param=np.unravel_index(np.argmax(accuracy), np.array(accuracy).shape)
print("Max paramters at index: ", max_param, "with accuracy: ", accuracy[max_param])

C_max = Cs[max_param[0]]
gamma_max = gammas[max_param[1]]
print("TRUE VALUES C: ", C_max, "Gamma: ", gamma_max)
print("LOG VALUES  C: ", np.log(C_max), "Gamma: ", np.log(gamma_max))

# retrain model with parameters from grid search and evaluate on validation set
# almost the same as before??
svm_model = SVC(C=C_max, gamma=gamma_max)
svm_model.fit(X_train, Y_train)
y_pred = svm_model.predict(X_validate)
print("Train accuracy: ", accuracy_score(svm_model.predict(X_train), Y_train))
print("Test accuracy: ", accuracy_score(y_pred, Y_test))
print("Support vectors: ", svm_model.n_support_)

# plot decision boundary
Z = svm_model.predict(np.c_[x1_axis.ravel(), x2_axis.ravel()])
Z = Z.reshape(x1_axis.shape)
fig = plt.subplot()
fig.contourf(x1_axis, x2_axis, Z)
fig.scatter(svm_model.support_vectors_[:,0], svm_model.support_vectors_[:,1])

# fit with adjusted gamma
svm_model=SVC(C=C_max, gamma=gamma_max/4)
svm_model.fit(X_train, Y_train)
y_pred=svm_model.predict(X_validate)
print("Train accuracy: ", accuracy_score(svm_model.predict(X_train), Y_train))
print("Test accuracy: ", accuracy_score(y_pred, Y_test))
print("Support vectors: ", svm_model.n_support_)

#plot decision boundary
Z = svm_model.predict(np.c_[x1_axis.ravel(), x2_axis.ravel()])
Z = Z.reshape(x1_axis.shape)
fig = plt.subplot()
fig.contourf(x1_axis, x2_axis, Z)
fig.scatter(svm_model.support_vectors_[:,0], svm_model.support_vectors_[:,1])


