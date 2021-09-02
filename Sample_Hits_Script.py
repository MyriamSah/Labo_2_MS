# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 12:35:17 2021

@author: Myriam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from minisom import MiniSom
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import pickle

with open('PSY3008', 'rb') as infile:
    som = pickle.load(infile)
    
X = pd.read_csv (r'C:\Users\Myriam\Excel_pr_python\Data_TRS_Ctrl.csv')
X = np.array(X)

  
Gr = pd.read_csv (r'C:\Users\Myriam\Excel_pr_python\Base_de_Données_M_S_SOM.csv')
Gr = np.array(Gr)
Gr = np.ravel(Gr)

Y = pd.read_csv(r'C:\Users\Myriam\Excel_pr_python\Base_de_Données_M_S_SOM_Python.csv')

# Try = som.labels_map(X,Gr)
# print('Try', Try)

i=0
W = som.get_weights()

S = np.arange(64)
S = S.reshape((8,8))
S = np.zeros_like(S)
#print('Ordre', X[0], X[1], X[2], X[3], X[4] )
for i in range(113):
      x = som.winner(X[i])
      S[x[0],x[1]]=S[x[0],x[1]]+1
      i= i+1

S = S.T
print("S", S)
print (W[5,1])
winner_coordinates = np.array([som.winner(x) for x in X])
print('winner_coordinates', winner_coordinates)

W = som.get_weights()
#print(W[0,0])

Q = som.quantization(X)
#print('Q', Q.shape)
QE = som.quantization_error(X)
#print('QE', QE)


fig, ax = plt.subplots()
im = ax.imshow(S, cmap="RdPu")

for i in range(8):
    for j in range(8):
        text = ax.text(j, i, S[i, j],
                        ha="center", va="center", color="k")
        
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel(ylabel="Sample Hits", rotation=-90, va="bottom")
ax.invert_yaxis()
plt.savefig('Sample_Hits.png', transparent=True)
plt.show()



