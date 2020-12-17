#!/usr/bin/env python
# coding: utf-8

#Training datasets
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import  matplotlib.pyplot as plt

minofsam=0
maxofsam=2
numofsam=30
X=np.random.uniform(minofsam,maxofsam,numofsam)
T=3*X
#plt.scatter(X,T)
#plt.scatter(X,T)
#plt.title("Correct  data",fontsize=25)
#plt.xlabel("Input(x)",fontsize=18)
#plt.ylabel("Output",fontsize=18)
#plt.show()

noise=np.random.normal(0,0.5,numofsam)
T=T+noise

#sampledata
plt.scatter(X,T)
plt.title("Correct  data",fontsize=25)
plt.xlabel("Input(x)",fontsize=18)
plt.ylabel("Output",fontsize=18)
plt.show()

#Learning Weights
def identity(x):
    return x
#Cost function
def square_sum(Y,T):
    return np.sum(np.square(Y-T))/2

w=0
eta=0.001
iteration=50
E_save=np.array([])
weight_save=np.array([])

#Learning
for i in range(iteration):
    #Forward propagation
    weight_save=np.append(weight_save,w)
    Y=identity(w*X)
    E=square_sum(Y,T)
    E_save=np.append(E_save,E)
    #Backward propagation
    #This is　Gradient descent
    dw=np.sum(X*(Y-T))
    w=w-eta*dw

#print("weight is {0:.2f}".format(w))
    
#Ploting the learning result
xline=np.linspace(0,2,5)
yline=w*xline

plt.scatter(X,T)
plt.plot(xline,yline,color="red")
plt.title("Learning result",fontsize=25)
plt.xlabel("Input(x)",fontsize=18)
plt.ylabel("Output",fontsize=18)
plt.show()

plt.plot(E_save)
plt.title("Loss Function's transition",fontsize=22)
plt.xlabel("Iteration number",fontsize=18)
plt.ylabel("Loss value(E)",fontsize=18)
plt.show()

weight_save2=np.array([])
E_save2=np.array([])

for w in np.linspace(0,6,40):
    weight_save2=np.append(weight_save2,w)
    Y=identity(w*X)
    E=square_sum(Y,T)
    E_save2=np.append(E_save2,E)
plt.scatter(weight_save,E_save,color="red",s=40)
plt.plot(weight_save2,E_save2)
plt.title("Loss Function's transition",fontsize=22)
plt.xlabel("weight(w)",fontsize=18)
plt.ylabel("Loss value(E)",fontsize=18)
plt.show()
