#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.random.seed(1)
colors=("red","blue","green","gray","orange")

#Sample data
numofsam=20
#Guissian noise
std_dv=0.7
radius=3
x_center=np.random.randn(numofsam,2)*std_dv
s=np.random.uniform(0,2*np.pi,numofsam)
x1=np.sin(s)*radius
x2=np.cos(s)*radius
x_circle=np.c_[x1,x2]
X=np.vstack((x_center,x_circle))
t_group1=np.tile(0,(numofsam))
t_group2=np.tile(1,(numofsam))
T_vector=np.hstack((t_group1,t_group2))
T=np.eye(len(np.unique(T_vector)))[T_vector]

#Learning Setting
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def cross_entropy(Y,T):
    return -np.sum(T*np.log(Y+1e-7))

dimention = 5
W1=np.random.randn(2,dimention)
W2=np.random.randn(dimention,2)
B1=np.random.randn(1,dimention)
B2=np.random.randn(1,2)
eta=0.001
iteration=2000
E_save=np.array([])

#Learning
for i in range(iteration):
    #Forward propergation
    H=sigmoid(np.dot(X,W1)+B1)
    Y=softmax(np.dot(H,W2)+B2)
    #--Back poropergation
    E=cross_entropy(Y,T)
    E_save=np.append(E_save,E)
    dW2=np.dot(H.T,Y-T)
    dB2=np.sum(Y-T,axis=0,keepdims=True)
    dW1=np.dot(X.T,H*(1-H)*np.dot(Y-T,W2.T))
    dB1=np.sum(H*(1-H)*np.dot(Y-T,W2.T), axis=0, keepdims=True)
    W1=W1-eta*dW1
    W2=W2-eta*dW2
    B1=B1-eta*dB1
    B2=B2-eta*dB2
#Grid Setting
x1_min=X[:,0].min()-1
x1_max=X[:,0].max()+1
x2_min=X[:,1].min()-1
x2_max=X[:,1].max()+1
x1_grid=np.linspace(x1_min,x1_max,100)                 
x2_grid=np.linspace(x2_min,x2_max,100)
xx,yy=np.meshgrid(x1_grid,x2_grid)
X_grid=np.c_[xx.reshape(-1),yy.reshape(-1)]
H_grid=sigmoid(np.dot(X_grid,W1)+B1)
Y_grid=softmax(np.dot(H_grid,W2)+B2)
Y_vector=np.argmax(Y_grid,axis=1)
Y=np.eye(len(np.unique(Y_vector)))[Y_vector]

#plot
for i in np.unique(T_vector):
    plt.scatter(x=X[T_vector==i,0],y=X[T_vector==i,1],s=10,c=colors[i])
for i in np.unique(Y_vector):
    plt.scatter(x=X_grid[Y_vector==i,0],y=X_grid[Y_vector==i,1],s=10,c=colors[i],alpha=0.1)
plt.title("Learning result",fontsize=25)
plt.xlabel("X1",fontsize=18)
plt.ylabel("X2",fontsize=18)
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.show()

plt.plot(E_save)
plt.title("Loss Function",fontsize=25)
plt.xlabel("Iteration Number",fontsize=18)
plt.ylabel("Loss Value(E)",fontsize=18)
plt.show()

