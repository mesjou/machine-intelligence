# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:48:32 2022

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:26:16 2022

@author: User
"""
#Exercise 1
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("C:/Users/User/Documents/BSE weitere Semester/Machine Learning/MI2/Ex1/pca-data-2d.dat")

#a) scatter plot of centered data
#centering
data_centered= data - np.mean(data, axis=0)
plt.scatter(data_centered[:,0],data_centered[:,1])

#b) pca
# eigenvalue decomposition of covariance matrix
cov =np.cov(data.transpose())
#eigenvalues lambda and right eigenvectors eig
lamb, eig = np.linalg.eig(cov)
#sort according to eigenvalues (highest first)
idx=np.argsort(lamb)
eig_sorted=eig[:,idx]

#check (see slides p. 15)
a_1=np.transpose(eig[:,0])@ data_centered[0,:]
a_2=np.transpose(eig[:,1])@ data_centered[0,:]
x_recovered=a_1 * eig[:,0] + a_2 * eig[:,1]
print(data_centered[0,:]-x_recovered) #check: should be 0

#b)
A_1= data_centered @ np.transpose(eig[:,0])
A_2= data_centered @ np.transpose(eig[:,1])
plt.scatter(A_1,A_2)
#alternativ:

#c)projections: why are these lines?
data_reconstructed_1 = np.array( [a_1 * eig[:,0] for a_1 in A_1 ])
plt.scatter(data_reconstructed_1[:,0],data_reconstructed_1[:,1])

data_reconstructed_2 = np.array([a_2 * eig[:,1] for a_2 in A_2 ])
plt.scatter(data_reconstructed_2[:,0],data_reconstructed_2[:,1])
#%% Excericise 3 

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import seaborn as sns

data = np.loadtxt("C:/Users/User/Documents/BSE weitere Semester/Machine Learning/MI2/Ex1/expDat.txt", delimiter=',', skiprows=1, usecols=range(1,21))

#a) find 20 PCs
pca=PCA(n_components=20)
pca.fit(data)
print(pca.components_)
pca_components=pca.components_

#b)i)scatter plot
A_1= data @ np.transpose(pca_components[:,0])
A_2= data @ np.transpose(pca_components[:,1])

fig, ax = plt.subplots()
scatter=ax.scatter(A_1, A_2, c=range(100))
legend1 = ax.legend(*scatter.legend_elements(num=10),
                    loc="upper left", title="time")
ax.add_artist(legend1)

#ii)line plot
plt.plot(range(100), A_1, label='PC1')
plt.plot(range(100), A_2, label='PC2')
plt.ylabel('Projection onto PCs')
plt.xlabel('time')
plt.legend()
plt.show()

#c) shuffle data
idx=list(range(20))
random.shuffle(idx)
data_shuffled=data[:, idx]
#covariance matrices: they are the same!
cov_data = np.cov(data)
cov_data_shuffled = np.cov(data_shuffled)
print(cov_data)
print(cov_data_shuffled)
sns.heatmap(cov_data)
sns.heatmap(cov_data_shuffled)

#scree plot: same matrices means also eig.val. are the same
eig=np.linalg.eigvals(cov_data)
eig_shuffled=np.linalg.eigvals(cov_data_shuffled)
plt.plot(range(100), eig, label='original data')
plt.plot(range(100), eig_shuffled, label='shuffled data')
plt.ylabel('eigenvalues')
plt.xlabel('time')
plt.legend()
plt.show()

#d) also shuffeling over time index would not change covariance matrix (?))