#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
#Parameter curve
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
#Plot color scheme library
import matplotlib.cm as cm

#Sampledata
numofsam=20

#Noise in Gaussian establishment distribution
std_dv=1.5

#Center of the data set
group1=np.array([2,2])
group2=np.array([-2,-2])

#Add noise
group1=group1+np.random.randn(numofsam,2)*std_dv
group2=group2+np.random.randn(numofsam,2)*std_dv

#Sequence concatenation
X=np.vstack((group1,group2))

#correct data
correct1=np.ones((numofsam,1))
correct2=np.zeros((numofsam,1))
T=np.vstack((correct1,correct2))

#plot
#plt.scatter(X[:,0],X[:,1],s=20,c=T[:,0],cmap=cm.bwr,vmin=0,vmax=1)
#plt.title("Correct Data",fontsize=25)
#plt.xlabel("X1",fontsize=18)
#plt.ylabel("X2",fontsize=18)
#plt.show()

#sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))
#cross-entropy error
def cross_entropy(Y,T):
    return -np.sum(T*np.log(Y+1e-7))

W=np.array([[-3],[-3]])
eta=0.01
iteration=10
E_save=np.array([])
w1_save=np.array([])
w2_save=np.array([])

#Learning
for i in range(iteration):
    #Forward propagation
    w1_save=np.append(w1_save,W[0,0])
    w2_save=np.append(w2_save,W[1,0])
    Y=sigmoid(np.dot(X,W))
    
    #Buckward propergation
    E=cross_entropy(Y,T)
    E_save=np.append(E_save,E)
    dW=np.sum(X*(Y-T),axis=0)
    dW=np.reshape(dW,(2,1))
    W=W-eta*dW

#print("Weight 1 is{0:2f}".format(W[0,0]))
#print("Weight 2 is{0:2f}".format(W[1,0]))

#plot the Learning result

#Grid Setting
x1_min = X[:,0].min()-1
x1_max = X[:,0].max()+1
x2_min = X[:,1].min()-1
x2_max = X[:,1].max()+1
x1_grid=np.linspace(x1_min,x1_max,100)
x2_grid=np.linspace(x2_min,x2_max,100)

xx,yy=np.meshgrid(x1_grid,x2_grid)
X_grid=np.c_[xx.reshape(-1),yy.reshape(-1)]

#Grid plot
plt.scatter(X_grid[:,0],X_grid[:,1],s=1)
plt.title("Learning result",fontsize=25)
plt.xlabel("X1",fontsize=18)
plt.ylabel("X2",fontsize=18)
plt.show()

#Learning result
Y_grid=sigmoid(np.dot(X_grid,W))
Y_predict=np.around(Y_grid)
plt.scatter(X[:,0],X[:,1],c=T[:,0],cmap=cm.bwr,s=20,vmin=0,vmax=1)
plt.scatter(X_grid[:,0],X_grid[:,1],c=Y_predict[:,0],cmap=cm.bwr,s=1,alpha=0.5,vmin=0,vmax=1)
plt.title("Learning result",fontsize=25)
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.show()

plt.plot(E_save)
plt.show()

E_save2=np.array([])
resolution=50
w1=np.linspace(-3,3,resolution)
w2=np.linspace(-3,3,resolution)
xx2,yy2=np.meshgrid(w1,w2)

for i in w1:
    for j in w2:
        W=np.array([[i],[j]])
        Y=sigmoid(np.dot(X,W))
        E=cross_entropy(Y,T)
        E_save2=np.append(E_save2,E)
E_save2=np.reshape(E_save2,(resolution,resolution))

#plot
fig=plt.figure(figsize=(8,5))
ax=Axes3D(fig)
ax.plot_surface(xx2,yy2,E_save2,cmap=cm.jet,vmin=E_save2.min(),vmax=E_save2.max(),shade=True,alpha=0.3)
ax.scatter(w1_save,w2_save,E_save,color="blue",s=40)

plt.show()

