#!/usr/bin/env python
# coding: utf-8

# In[12]:


#One-hot classification
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
colors=("red","blue","green","gray","orange")
numofsam=20
#Gaussian noise
std_dv=0.6
group1=np.array([2.5,2.5])
group2=np.array([7.5,7.5])
group3=np.array([0,10])
group4=np.array([6,15])
group5=np.array([4,8])
#add noise
group1=group1+np.random.randn(numofsam,2)*std_dv
group2=group2+np.random.randn(numofsam,2)*std_dv
group3=group3+np.random.randn(numofsam,2)*std_dv
group4=group4+np.random.randn(numofsam,2)*std_dv
group5=group5+np.random.randn(numofsam,2)*std_dv
X = np.vstack((group1,group2,group3,group4,group5))
#correct data
t_group1=np.tile(0,(numofsam))
t_group2=np.tile(1,(numofsam))
t_group3=np.tile(2,(numofsam))
t_group4=np.tile(3,(numofsam))
t_group5=np.tile(4,(numofsam))
T_vector=np.hstack((t_group1,t_group2,t_group3,t_group4,t_group5))
T=np.eye(len(np.unique(T_vector)))[T_vector]
#Learning Setting
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

def cross_entropy(Y,T):
    return -np.sum(T*np.log(Y+1e-7))

W=np.random.randn(2,5)
B=np.random.randn(1,5)
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
x2_grid = np.linspace(x2_min,x2_max,100)
xx,yy=np.meshgrid(x1_grid,x2_grid)
X_grid=np.c_[xx.reshape(-1),yy.reshape(-1)]
Y_grid=softmax(np.dot(X_grid,W)+B)
Y_grid_vector=np.argmax(Y_grid,axis=1)
Y=np.eye(len(np.unique(T_vector)))[Y_grid_vector]

#Plot
for i in np.unique(T_vector):
    plt.scatter(x=X[T_vector==i,0],y=X[T_vector==i,1],s=10,c=colors[i])
for i in np.unique(T_vector):
    plt.scatter(x=X_grid[Y_grid_vector==i,0],y=X_grid[Y_grid_vector==i,1],s=10,c=colors[i],alpha=0.1)
plt.title("Learning result",fontsize=25)
plt.xlabel("X1",fontsize=18)
plt.ylabel("X2",fontsize=18)
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.show()

#Loss result
plt.plot(E_save)
plt.title("Loss Function",fontsize=25)
plt.xlabel("Iteration Number",fontsize=18)
plt.ylabel("Loss Value(E)",fontsize=18)
plt.ylim(0,50)
plt.show()

