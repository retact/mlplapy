#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
numofsam=20 

#Gaussian noise
std_dv=0.5
group1=np.array([2.5,2.5])
group2=np.array([5,5])
#print(group1)
#print(group2)
group1=group1+np.random.randn(numofsam,2)*std_dv
group2=group2+np.random.randn(numofsam,2)*std_dv
#print("group1 \n",group1)
#print("group2 \n",group2)
X=np.vstack((group1,group2))
#print('X is \n',X)
t_group1=np.zeros((numofsam,1))
t_group2=np.ones((numofsam,1))
#print("t_group is \n",t_group1)
#print("t_group2 is \n",t_group2)

T=np.vstack((t_group1,t_group2))
#print("X is \n",X)
#print("X is \n",X[:,0])
#print("X is \n",X[:,1])
#plt.scatter(X[:,0],X[:,1],s=20,c=T[:,0],cmap=cm.bwr)
#plt.title("Correct Data",fontsize=25)
#plt.xlabel("X1",fontsize=18)
#plt.ylabel("X2",fontsize=18)
#plt.show()

#Learning Weights and Bias
def sigmoid(x):
   return 1/(1+np.exp(-x))

def cross_entropy(Y,T):
   return -np.sum(T*np.log(Y+1e-7))

#create randn Weights and Biases
W=np.random.randn(2,1)
B=np.random.randn(1,1)

print("randn Weight are \n",W)
print("randn Bias are \n",B)

eta=0.001
iteration=10000
E_save=np.array([])

#Learning
for i in range(iteration):
   #Forward propergation
   Y=sigmoid(np.dot(X,W)+B)
   #print("Y number ",i,Y,sep='\n')
   E=cross_entropy(Y,T)
   E_save=np.append(E_save,E)
   dW=np.sum(X*(Y-T),axis=0)
   dW=np.reshape(dW,(2,1))
   dB=np.sum(Y-T)
   W=W-eta*dW
   B=B-eta*dB
   
print("w1 bias is{0:.2f}".format(W[0,0]))
print("w2 bias is{0:.2f}".format(W[1,0]))
print("bias is {0:.2f}".format(B[0,0]))

#grid Setting
x1_min=X[:,0].min()-1
x1_max=X[:,0].max()+1
x2_min=X[:,1].min()-1
x2_max=X[:,1].max()+1
x1_grid=np.linspace(x1_min,x1_max,100)
x2_grid=np.linspace(x2_min,x2_max,100)
xx,yy=np.meshgrid(x1_grid,x2_grid)
X_grid=np.c_[xx.reshape(-1),yy.reshape(-1)]
#Learning Result Setting
Y_grid=sigmoid(np.dot(X_grid,W)+B)

#onehot
Y_predict=np.around(Y_grid)
plt.scatter(X[:,0], X[:,1], c=T[:,0], cmap=cm.bwr, s=20)

plt.scatter(X_grid[:,0], X_grid[:,1],c=Y_predict[:,0], cmap=cm.bwr, s=1,alpha=0.5)
plt.title("Learning result",fontsize=25)
plt.xlabel("X1",fontsize=18)
plt.ylabel("X2",fontsize=18)
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.show()

#Loss Lapses
plt.plot(E_save)
plt.title("Loss Function",fontsize=25)
plt.xlabel("Iteration Number",fontsize=18)
plt.ylabel("Loss Value(E)",fontsize=18)
plt.show()

