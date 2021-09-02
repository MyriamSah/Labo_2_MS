# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 20:09:24 2021

@author: Myriam
"""

from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import pickle

with open('PSY3008', 'rb') as infile:
    som = pickle.load(infile)
 
X = pd.read_csv (r'C:\Users\Myriam\Excel_pr_python\Data_TRS_Ctrl.csv')
X = np.array(X)

W = som.get_weights()


N = np.reshape(W,[64,15])

agglom = AgglomerativeClustering(n_clusters=2,linkage='complete').fit(N)



Lab=agglom.labels_
print('Lab', Lab)

# CoefS = metrics.silhouette_score(N, Lab, metric='euclidean')

# print(CoefS)


dist = distance_matrix(N,N)

Z = hierarchy.linkage(dist, 'complete')

plt.figure(figsize=(18, 50))
dendro = hierarchy.dendrogram(Z, leaf_rotation=0, leaf_font_size=30, orientation='right')


