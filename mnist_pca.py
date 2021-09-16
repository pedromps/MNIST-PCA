# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

data_csv = pd.read_csv(os.path.join(os.getcwd(), "mnist_train.csv"))
data = np.array(data_csv)

#loop to check all possibilities from 1 single feature to #features-1
for i in range(1, data.shape[1]-1):
    #if the PCA has a good explained variance ratio, we can reduce dimensions without losing much info
    pca = PCA(copy=True, iterated_power='auto', n_components=i, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
    pca_input = data
    pca = pca.fit(pca_input) 
    print("Explained Variance of "+str(i)+" adds up to {:.2f}%".format(100*np.sum(pca.explained_variance_ratio_)))
    #Z_pca = pca.transform(pca_input) <- commented as it wont be used