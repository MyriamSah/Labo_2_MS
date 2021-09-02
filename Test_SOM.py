# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from minisom import MiniSom
import plotly.graph_objects as go
import pickle
fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
#fig.write_html('Quantization_Error.html', auto_open=True)


#columns = ['FPT_U1', 'FPT_U2', 'FPT_U3', 'FPT_U4', 'FPT_U5', 'FPT_EG1', 'FPT_EG2', 'FPT_EG3', 'FPT_EG4', 'FPT_EG5', 'FPT_S1', 'FPT_S2', 'FPT_S3', 'FPT_S4', 'FPT_S5']
#data= list(csv.reader(open(r'C:\Users\Myriam\Test_Data.csv')))

X = pd.read_csv (r'C:\Users\Myriam\Excel_pr_python\Data_TRS_Ctrl.csv')
X = np.array(X)
#X = np.asmatrix(X)
#print(X)
#X = np.array(X)
#print(X.shape)
#X = np.reshape(X, (1,50))
#X = np.array_split(X, 50)
Gr = pd.read_csv (r'C:\Users\Myriam\Excel_pr_python\Base_de_Données_M_S_SOM.csv')
Gr = np.array(Gr)
Gr = np.ravel(Gr)
Y = pd.read_csv(r'C:\Users\Myriam\Excel_pr_python\Base_de_Données_M_S_SOM_Python.csv')
Y = np.array(Y)
#print(Y[0])

som = MiniSom(8,8,15, sigma=0.8, learning_rate=0.5, neighborhood_function='gaussian', activation_distance='euclidean')
Res= som.train_random(X, 100000, verbose=True)
#Res= som.train(X, 1000000, random_order=True, verbose=True)

#À enlever
#PL = np.array([som.winner(x) for x in X])   
winner_coordinates = np.array([som.winner(x) for x in X]).T
#print('winner_coordinates', PL[13], PL[17], PL[22], PL[25], PL[29], PL[30], PL[31], PL[32], PL[121], PL[122], PL[123], PL[124], PL[125])
cluster_index = np.ravel_multi_index(winner_coordinates, (8,8))

# Tab = []
# i = 0
# for i in range(126):
#     PL = np.array([som.winner(x) for x in X])
#     Tab = Tab + [Y[i],PL[i]]
#     i= i+1
# print('Tab', Tab)

W = som.get_weights()
#print('W', W[0,0])
Q = som.quantization(X)
#print('Q', Q.shape)
QE = som.quantization_error(X)
#print('QE', QE)
# Act= som.activation_response(X)
# print('activation', Act)
# Try = som.labels_map(X,Gr)
# print('Try', Try)


#imprime figure avec clusters et centroide

for c in np.unique(cluster_index):
    plt.scatter(X[cluster_index == c, 0],
                X[cluster_index == c, 1], label='cluster='+str(c), alpha=1.0)
    #print(c)
# # # for centroid in som.get_weights():
# # #     plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
# # #                 s=80, linewidths=15, color='k', label='centroid')
plt.legend();
plt.show()


with open('PSY3008','wb') as outfile:
    pickle.dump(som,outfile)






