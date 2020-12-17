#!/usr/bin/env python
# coding: utf-8

#Preparing dataSets
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
minofsam=0
maxofsam=2
numofsam=20
X=np.random.uniform(minofsam,maxofsam,numofsam)
T=X*3+5

#noise
noise=np.random.normal(0,0.4,numofsam)
T=T+noise

#plt.scatter(X,T)
#plt.title("Correct data",fontsize=20)
#plt.xlabel("Input",fontsize=18)
#plt.ylabel("Output",fontsize=18)
#plt.ylim(0,15)
#plt.show()

#Learning Weights and Biases
def identity(x):
    return x

def square_sum(Y,T):
    return  np.sum(np.square(Y-T))/2

w=0
b=0
eta=0.01
iteration=10

E_save = np.array([])

#Learning
for i in range(iteration):
    #Forward propagation
    Y=identity(w*X+b)
    E=square_sum(Y,T)
    E_save=np.append(E_save,E)
    
    #backward propagation
    dw=np.sum(X*(Y-T))
    w=w-eta*dw
    db=np.sum(Y-T)
    b=b-eta*db
#print("Weight is {0:.2f}".format(w))
#print("Bias  is  {0:.2f}".format(b))

#plot
x_line = np.linspace(0,2,10)
y_line = w*x_line+b
plt.scatter(X,T)
plt.plot(x_line,y_line,color="red")
plt.title("Learning result",fontsize=20)
plt.xlabel("Input(x)",fontsize=18)
plt.ylabel("Output(y)",fontsize=18)
plt.ylim(0,12)
plt.show()

plt.plot(E_save)
plt.title("Loss Function's transition",fontsize=20)
plt.xlabel("Iteration numer",fontsize=18)
plt.ylabel("Loss value(E)",fontsize=18)
plt.show()
