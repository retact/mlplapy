#!/usr/bin/env python
# coding: utf-8

#Onehot　binary classification
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
colors=("red","blue","lightgreen","gray","cyan")
numofsam=20
std_dv=1
group1=np.array([2.5,2.5])
group2=np.array([7.5,7.5])
#print("1group",group1)
group1=group1+np.random.randn(numofsam,2)*std_dv
group2=group2+np.random.randn(numofsam,2)*std_dv
#print("2 group",group1)
X=np.vstack((group1,group2))
#print("X is",X)
t_group1=np.tile(0,(numofsam))
t_group2=np.tile(1,(numofsam))
T_vector=np.hstack((t_group1,t_group2))
T=np.eye(len(np.unique(T_vector)))[T_vector]

#Learning Setting
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

def cross_entropy(Y,T):
    return -np.sum(T*np.log(Y+1e-7))

W=np.random.randn(2,2)
B=np.random.randn(1,2)
eta=0.001
iteration=10000
E_save=np.array([])

#Learning
for i in range(iteration):
    Y=softmax(np.dot(X,W)+B)
    E=cross_entropy(Y,T)
    E_save=np.append(E_save,E)
    dW=X.T.dot(Y-T)
    dB=np.sum(Y-T,axis=0,keepdims=True)
    W=W-eta*dW
    B=B-eta*dB

#Grid Setting
x1_min=X[:,0].min()-1
x1_max=X[:,0].max()+1
x2_min=X[:,1].min()-1
x2_max=X[:,1].max()+1
x1_grid=np.linspace(x1_min,x1_max,100)
x2_grid=np.linspace(x2_min,x2_max,100)
xx,yy=np.meshgrid(x1_grid,x2_grid)
X_grid=np.c_[xx.reshape(-1),yy.reshape(-1)]

#Result ploting
Y_grid=softmax(np.dot(X_grid,W)+B)
Y_grid_vector=np.argmax(Y_grid,axis=1)
Y=np.eye(len(np.unique(T_vector)))[Y_grid_vector]

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

#Loss Function
plt.plot(E_save)
plt.title("Loss Function",fontsize=25)
plt.xlabel("Iteration Number",fontsize=18)
plt.ylabel("Loss Value(E)",fontsize=18)
plt.show()
