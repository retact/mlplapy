#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn.datasets as datasets

colors=("red","blue","lightgreen","gray","cyan")

#Import datasets
iris=datasets.load_iris()
input_data=iris.data
ave=np.average(input_data,axis=0)
std=np.std(input_data,axis=0)
input_data=(input_data-std)/ave
#print("iris is ",iris)
#print("input data is ",input_data)
#print ("ave is",ave)
#print("std is",std)
X=iris.data[0:150:2,:2]
X_test=iris.data[1:150:2,:2]
X=input_data[0:150:2,:2]
X_test=input_data[1:150:2,:2]

#correct data
T_vector=iris.target[0:150:2]
T_test_vector=iris.target[1:150:2]
T=np.eye(len(np.unique(T_vector)))[T_vector]
T_test=np.eye(len(np.unique(T_vector)))[T_test_vector]

#Learning Setting
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

def cross_entropy(Y,T):
    return -np.sum(T*np.log(Y+1e-7))

W=np.random.randn(2,len(np.unique(T_vector)))
B=np.random.randn(1,len(np.unique(T_vector)))
eta=0.001
iteration=10000
E_save=np.array([])
E_save_test=np.array([])

#Learning
for i in range(iteration):
    #Foward propergation
    Y=softmax(np.dot(X,W)+B)
    Y_test=softmax(np.dot(X_test,W)+B)
    #Back propergation
    E=cross_entropy(Y,T)
    E_test=cross_entropy(Y_test,T)
    E_save=np.append(E_save,E)
    E_save_test=np.append(E_save_test,E_test)
    dW=X.T.dot(Y-T)
    dB=np.sum(Y-T,axis=0,keepdims=True)
    W=W-eta*dW
    B=B-eta*dB

#Grid Setting
x1_min = X[:,0].min()-1
x1_max = X[:,0].max()+1
x2_min = X[:,1].min()-1
x2_max = X[:,1].max()+1
x1_grid = np.linspace(x1_min,x1_max,100)     
x2_grid = np.linspace(x2_min,x2_max,100)
xx,yy = np.meshgrid(x1_grid,x2_grid)
X_grid = np.c_[xx.reshape(-1),yy.reshape(-1)]
Y_grid = softmax(np.dot(X_grid, W)+B)
Y_test_data = softmax(np.dot(X_test, W)+B)
Y_grid_vector = np.argmax(Y_grid,axis=1)
Y_test_vector = np.argmax(Y_test_data,axis=1)

for i in np.unique(T_vector):
    plt.scatter(x=X[T_vector==i,0],y=X[T_vector==i,1],s=10,c=colors[i])
for i in np.unique(T_vector):
    plt.scatter(x=X_grid[Y_grid_vector==i,0],y=X_grid[Y_grid_vector==i,1],s=10,c=colors[i],alpha=0.2)
plt.title("Learning result",fontsize=25)
plt.xlabel("X1",fontsize=18)
plt.ylabel("X2",fontsize=18)
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.show()

#Loss function
plt.plot(E_save)
plt.plot(E_save_test)
plt.title("Loss Function",fontsize=25)
plt.xlabel("Iteration Number",fontsize=18)
plt.ylabel("Loss Value(E)",fontsize=18)
plt.show()

