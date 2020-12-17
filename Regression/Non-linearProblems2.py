#!/usr/bin/env python
# coding: utf-8

# Momentum　method
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
#Seed 　Vlue
np.random.seed(1)
numofsam=30
minofsam=0
maxofsam=2*np.pi
X=np.random.uniform(minofsam,maxofsam,numofsam)
#noise
noise=np.random.normal(0,0.1,numofsam)
T=np.sin(X)+noise
plt.scatter(X,T,s=5)
plt.title("Correct data",fontsize=25)
plt.xlabel("Input(x)",fontsize=18)
plt.ylabel("Output(y)",fontsize=18)
plt.show()

#Learning Setting
def identity(x):
    return x
def sigmoid(x):
    return 1/(1+np.exp(-x))
def squaresum(Y,T):
    return np.sum(np.square(Y-T))/2
dimention=10
W1=np.random.randn(dimention,1)
W2=np.random.randn(1,dimention)
#print(W1)
#print(W2)
B1=np.random.randn(dimention,1)
B2=np.random.randn(1,1)
W1_vel=np.zeros_like(W1)
W2_vel=np.zeros_like(W2)
B1_vel=np.zeros_like(B1)
B2_vel=np.zeros_like(B2)
alpha=0.9
eta=0.005
E_save=np.array([])
iteration=1000

#Learning
for i in range(iteration):
    #forward propergatiion
    H=sigmoid(X*W1+B1)
    Y=identity(np.dot(W2,H)+B2)
    #backward propergation
    E=squaresum(Y,T)
    E_save=np.append(E_save,E)
    dW2=np.sum(H*(Y-T),axis=1)
    dB2=np.sum(Y-T)
    dW1=W2*np.sum(X*H*(1-H)*(Y-T),axis=1)
    dB1=W2*np.sum(H*(1-H)*(Y-T),axis=1)
    #Save speed
    W1_vel=-eta*dW1.T+alpha*W1_vel
    W2_vel=-eta*dW2+alpha*W2_vel
    B1_vel = -eta*dB1.T+alpha*B1_vel
    B2_vel = -eta*dB2+alpha*B2_vel
    #print("dW1 is",dW1)
    #print("T is",dW1.T)
    W1 = W1+W1_vel
    W2 = W2+W2_vel
    B1 = B1+B1_vel
    B2 = B2+B2_vel
    
X_line=np.linspace(0*np.pi,maxofsam+0*np.pi,200)
H_line=sigmoid(X_line*W1+B1)
Y_line=np.ravel(np.dot(W2,H_line)+B2)
    
plt.scatter(X, T,s=5,label="Correct Data")
plt.plot(X_line, Y_line,"red",label="Learning Result")
plt.title("Learning Result",fontsize=25)
plt.xlabel("Input(x)",fontsize=18)
plt.ylabel("Output(y)",fontsize=18)
plt.legend(fontsize=8)
plt.show()

plt.plot(E_save)
plt.xlim(0,200)
plt.ylim(0,10)
plt.show()

