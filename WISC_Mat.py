# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:52:46 2021

@author: Myriam
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

with open('PSY3008', 'rb') as infile:
    som = pickle.load(infile)
    
X = pd.read_csv (r'C:\Users\Myriam\Excel_pr_python\Data_TRS_Ctrl.csv')
X = np.array(X)
A = pd.read_csv (r'C:\Users\Myriam\Excel_pr_python\WISC_Mat.csv')
A = np.array(A)
  
Gr = pd.read_csv (r'C:\Users\Myriam\Excel_pr_python\Base_de_Données_M_S_SOM.csv')
Gr = np.array(Gr)
Gr = np.ravel(Gr)

Y = pd.read_csv(r'C:\Users\Myriam\Excel_pr_python\Base_de_Données_M_S_SOM_Python.csv')

# Try = som.labels_map(X,Gr)
# print('Try', Try)

i=0
BMU=[]
PX=[]
PY=[]
P= np.arange(64)
P=P.reshape((8,8))
P=np.zeros_like(P)
S = np.arange(64)
S = S.reshape((8,8))
S = np.zeros_like(P)
#print('Ordre', X[0], X[1], X[2], X[3], X[4] )
for i in range(113):
      x = som.winner(X[i])
      if A[i] == 999:
          S[x[0],x[1]]=S[x[0],x[1]]
          P[x[0],x[1]]=P[x[0],x[1]]
      else:
          S[x[0],x[1]]=S[x[0],x[1]]+1
          P[x[0],x[1]]=P[x[0],x[1]]+A[i]
      i= i+1


for i in range(8):
    for j in range(8):
        P[i, j] = int(P[i, j])/int(S[i, j])
        j = j+1
    i = i+1
P = P.T
winner_coordinates = np.array([som.winner(x) for x in X])
#print('winner_coordinates', winner_coordinates)

W = som.get_weights()
#print(W[0,0])

Q = som.quantization(X)
#print('Q', Q.shape)
QE = som.quantization_error(X)
#print('QE', QE)


fig, ax = plt.subplots()
im = ax.imshow(P, cmap="rainbow_r")

for i in range(8):
    for j in range(8):
        if P[i, j] == 0:
            text = ax.text(j, i, 'N/A',
                        ha="center", va="center", color="k")
        else:
            text = ax.text(j, i, P[i, j],
                        ha="center", va="center", color="k")
        
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel(ylabel="'WISC-Mat", rotation=-90, va="bottom")
# # # Z=[400,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]

# # # plt.scatter(PX,PY, s=Z)
# plt.xlim(-0.5,7.5)
# plt.ylim(-0.5,7.5)

plt.show()