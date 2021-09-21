# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data_csv = pd.read_csv(os.path.join(os.getcwd(), "mnist_train.csv"))
data_csv.drop("5", inplace = True, axis = 1)
data = np.array(data_csv, dtype = np.float64)

# normalising data
data/=255

variances = np.zeros(data.shape[1]-2)
# loop to check all possibilities from 1 single feature to #features-1
for i in range(1, data.shape[1]-1):
    # if the PCA has a good explained variance ratio, we can reduce dimensions without losing much info
    pca = PCA(copy = True, iterated_power = 'auto', n_components = i, random_state = None, svd_solver = 'auto', tol = 0.0, whiten = False)
    pca_input = data
    pca = pca.fit(pca_input) 
    print("Explained Variance of " + str(i) + " adds up to {:.2f}%".format(100*np.sum(pca.explained_variance_ratio_)))
    # Z_pca = pca.transform(pca_input) <- commented as it wont be used
    variances[i-1] = 100*np.sum(pca.explained_variance_ratio_)
  
plt.figure()
plt.semilogx(variances)
plt.xlabel("Components")
plt.ylabel("Explained Variance in %")
plt.grid()
plt.title("Explained Variance in % for PCAs with different amounts of components")
