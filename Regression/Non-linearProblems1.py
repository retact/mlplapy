#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
numofsam=30
minofsam=0
maxofsam=2*np.pi
X=np.random.uniform(minofsam,maxofsam,numofsam)

#noise
#Random numbers according to a normal distribution
noise=np.random.normal(0,0.1,numofsam)
T=np.sin(X)+noise

#plot
plt.scatter(X,T,s=5)
plt.show()

def identity(x):
    return x

def  sigmoid(x):
    return 1/(1+np.exp(-x))

def square_sum(Y,T):
    return np.sum(np.square(Y-T))/2

#Number of neurons in the hidden layer
dimention=50

#Average 0 standard deviation1
W1=np.random.randn(dimention,1)
W2=np.random.randn(1,dimention)
B1=np.random.randn(dimention,1)
B2=np.random.randn(1,1)

#Generate 0 array
W1_vel=np.zeros_like(W1)
W2_vel = np.zeros_like(W2)
B1_vel = np.zeros_like(B1)
B2_vel = np.zeros_like(B2)
eta=0.0001
alpha=0.9

E_save=np.array([])
iteration=5000

#Learning
for i in range(iteration):
    #forward propagation
    H=sigmoid(X*W1+B1)
    Y=identity(np.dot(W2,H)+B2)
    E=square_sum(Y,T)
    E_save=np.append(E_save,E)
    
    dW2=np.sum(H*(Y-T),axis=1)
    dB2=np.sum(Y-T)
    dW1=W2*np.sum(X*H*(1-H)*(Y-T),axis=1)
    dB1 = W2*np.sum(H*(1-H)*(Y-T))
    
    W1_vel = -eta*dW1.T+alpha*W1_vel
    W2_vel = -eta*dW2+alpha*W2_vel
    B1_vel = -eta*dB1.T+alpha*B1_vel
    B2_vel = -eta*dB2+alpha*B2_vel
    
    #back propagation
    W1 = W1+W1_vel
    W2 = W2+W2_vel
    B1 = B1+B1_vel
    B2 = B2+B2_vel

X_line=np.linspace(0*np.pi,maxofsam+0*np.pi,200)
H_line=sigmoid(X_line*W1+B1)
Y_line=np.ravel(np.dot(W2,H_line)+B2)
#scatter diagram
plt.scatter(X,T,s=5)
plt.plot(X_line,Y_line,"red")
plt.title("LearningResult",fontsize=25)
plt.show()

plt.plot(E_save)
plt.xlim(0,200)
plt.ylim(0,10)
plt.show()
