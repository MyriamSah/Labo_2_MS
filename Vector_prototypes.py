# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:02:26 2021

@author: Myriam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pickle

with open('PSY3008', 'rb') as infile:
    som = pickle.load(infile)
    
X = pd.read_csv (r'C:\Users\Myriam\Excel_pr_python\Data_TRS_Ctrl.csv')
X = np.array(X)
    
Gr = pd.read_csv (r'C:\Users\Myriam\Excel_pr_python\Base_de_Données_M_S_SOM.csv')
Gr = np.array(Gr)
Gr = np.ravel(Gr)

Y = pd.read_csv(r'C:\Users\Myriam\Excel_pr_python\Base_de_Données_M_S_SOM_Python.csv')

W = som.get_weights()
print(W[0,0])

# Clu1 = (W[1,6]+W[0,6]+W[3,7]+W[1,7]+W[0,7]+W[2,7]+W[2,6]+W[1,5])/8
# Clu2 = (W[0,0]+W[0,1]+W[0,2]+W[0,3]+W[0,4]+W[0,5]+W[1,0]+W[1,1]+W[1,2]+W[1,3]+W[1,4]+W[2,0]+W[2,1]+W[2,2]
#         +W[2,3]+W[2,4]+W[2,5]+W[3,0]+W[3,1]+W[3,2]+W[3,3]+W[3,4]+W[3,5]+W[3,6]+W[4,0]+W[4,1]+W[4,2]+W[4,3]+W[4,4]
#         +W[4,5]+W[4,6]+W[4,7]+W[5,0]+W[5,1]+W[5,2]+W[5,3]+W[5,4]+W[5,5]+W[5,6]+W[5,7]+W[6,0]+W[6,1]+W[6,2]+W[6,3]
#         +W[6,4]+W[6,5]+W[6,6]+W[6,7]+W[7,0]+W[7,1]+W[7,2]+W[7,3]+W[7,4]+W[7,5]+W[7,6]+W[7,7])/56
Red = (W[1,0]+W[1,1]+W[1,2]+W[2,0]+W[2,1]+W[2,2]+W[3,0]+W[3,1]+W[3,2]+W[3,3]+W[4,0]+W[4,1]+W[4,2]+W[4,3]
            +W[4,4]+W[4,5]+W[5,0]+W[5,1]+W[5,2]+W[5,3]+W[5,4]+W[6,0]+W[6,1]+W[6,2]+W[6,3]+W[6,4]+W[7,0]+W[7,1]
            +W[7,2]+W[7,3]+W[7,4]+W[7,5])/32
Green = (W[0,0]+W[0,1]+W[0,2]+W[0,3]+W[0,4]+W[0,5]+W[1,3]+W[1,4]+W[2,3]+W[2,4]+W[2,5]+W[3,4]+W[3,5]+W[3,6]
            +W[4,6]+W[4,7]+W[5,5]+W[5,6]+W[5,7]+W[6,5]+W[6,6]+W[6,7]+W[7,6]+W[7,7])/24

    
fig, ax = plt.subplots()
ax.plot(Red[0:5], label="Number of designs")
ax.plot(Red[5:10], label="Number of repetitions")
ax.plot(Red[10:15], label="Number of strategies")
ax.set_xlim(0,4)
ax.set_ylim(0,20)
ax.set_title("Patron Cluster rouge")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(Green[0:5], label="Number of designs")
ax.plot(Green[5:10], label="Number of repetitions")
ax.plot(Green[10:15], label="Number of strategies")
ax.set_xlim(0,4)
ax.set_ylim(0,20)
ax.set_title("Patron Cluster vert")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(W[0,6,0:5], label="Nombre de dessins")
ax.plot(W[0,6,5:10], label="Nombre de répétitions")
ax.plot(W[0,6,10:15], label="Nombre de stratégies")
ax.set_xlim(0,4)
ax.set_ylim(0,20)
ax.set_title("Vecteurs prototypes (0,6)")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(W[7,7,0:5], label="Nombre de dessins")
ax.plot(W[7,7,5:10], label="Nombre de répétitions")
ax.plot(W[7,7,10:15], label="Nombre de stratégies")
ax.set_xlim(0,4)
ax.set_ylim(0,20)
ax.set_title("Vecteurs prototypes (7,7)")
ax.legend()
plt.show()

DE = distance.euclidean(W[7,6],W[7,7])
print(DE)
P= np.arange(225)
P=P.reshape((15,15))
P=np.zeros_like(P)


# for i in range(15):
#     for j in range(15):
#         if 
Map = som.distance_map()

fig, ax = plt.subplots()
im = ax.imshow(Map, cmap="binary")
        
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel(ylabel="U-matrix", rotation=-90, va="bottom")
ax.invert_yaxis()

plt.savefig('U-matrix.png', transparent=True)

Dist = som._distance_from_weights(X)
print('Dist', Dist.shape)

