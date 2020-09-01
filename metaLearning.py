# -*- coding: utf-8 -*-
from pathlib import Path
import random
import numpy as np
import json
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge, LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor

path = Path('./')

#randomness goes from 0 to infinity
def generateTrainingData(nx=3):
    #nx=3
    length = int(np.random.uniform(30, 40000))
    freqs = [int(v) for v in np.clip(np.abs(np.random.normal(10, np.random.uniform(10, 100), size=nx)), 1, length)]
    X = []
    
    for i in range(nx):
        print('generating', i, 'out of', nx, 'inputs:','freq',freqs[i], 'length',length)
        sigma = random.choice([3,4,7,10,20])
        period1 = np.abs(np.random.normal(length, length/4))
        phase1 = length*2*random.random() 
        
        period2 = np.abs(np.random.normal(length/2, length/8))
        phase2 = length*random.random() 
        
        period3 = np.abs(np.random.normal(length/4, length/30))
        phase3 = length*0.5*random.random() 
        
        typical = np.random.uniform(0, 10000)
        rnge = [np.random.uniform(0, typical), np.random.uniform(typical, typical*5)]
        
        timeSeries = []       
        freq = freqs[i]
        times = [j*freq for j in range(int(length/freq))]
        
        if random.random() > 0.5:
            for t in range(length):
                x = np.abs(np.round(np.clip(np.random.normal(typical, typical*0.5/sigma), min(rnge), max(rnge)),2))
                d = (t, x)
                timeSeries.append(d)
        else:
            for t in range(length):
                x = np.round(np.clip(np.random.normal(typical, typical*0.5/sigma) + np.random.normal(typical*0.3, typical*0.03)*(np.sin(2*np.pi*t/period1 + phase1)) + np.random.normal(typical*0.1, typical*0.01)*(np.sin(2*np.pi*t/period2 + phase2)) + np.random.normal(typical*0.05, typical*0.001)*(np.sin(2*np.pi*t/period3 + phase3)), min(rnge), max(rnge)),2)
                d = (t, x)
                timeSeries.append(d)
        X.append(timeSeries)
    
    inXs = []
    for i, t in enumerate(range(length)):
        xs = [x[i][1] for x in X]
        inXs.append(xs)
    
    print('Normalizing inputs and output...')
    
    miu = np.random.uniform(0, 10000)
    sigma = np.random.normal(miu, miu/random.choice([3,4,7,10,20]))
    
    mius = np.mean(inXs, axis=0)
    sigmas = np.std(inXs, axis=0) + mius/10
    
    #print(mius, sigmas)
    
    print('Randomizing parameters for underlying relationship...')
    params1 = np.random.uniform(-1,1,len(X))
    params2 = np.random.uniform(-1,1,len(X))
    params3 = np.random.uniform(-1,1,2)
    zs = []
    print('Generating underlying relationship...')
    for dx in inXs:
        z1 = np.tanh(np.dot(params1, (dx-mius)/sigmas))
        z2 = np.tanh(np.dot(params2, (dx-mius)/sigmas))
        z3 = np.array([z1,z2])
        z = np.tanh(np.dot(params3, z3))
        zs.append(z)
    zs = np.array(zs)
    #print('z',zs)
    print(sigma)
    a = zs*sigma
    b = np.random.normal(miu, miu/10)
    resY = np.abs(a+b)

    print('Embedding generated relationship back into original time series...')
    currentY = resY[0]
    outY = []
    for i, t in enumerate(range(length)):
        outY.append([t,resY[i]])
        currentY = resY[i]
       
    print('Simulating sampling frequencies...')
    #freqs of X
    outX = []
    for i, freq in enumerate(freqs):
        times = [j*freq for j in range(int(length/freq))]
        xs = []
        for t, x in X[i]:
            if t in times:
                xs.append([t,x])
        outX.append(xs)
    
    #freq of Y
    freq = int(np.clip(np.abs(np.random.normal(10, np.random.uniform(10, 100))), 1, length))
    times = [j*freq for j in range(int(length/freq))]
    ys = []
    for t, y in outY:
        if t in times:
            ys.append([t,y])
    
    return outX, ys
    
def train(X, y):
    print("Pulling inputs and output data from time series...")
    #maxT = max([x[-1][0] for x in X])
    
    print('Synchronizing multiple time series...')
    T = [] 
    for x in X:
        T += [d[0] for d in x]
    
    T += [d[0] for d in y]
    T = sorted(list(set(T))) #common timestamp in X
    print(max(T))
    maxT = max(T)
    #train-test config
    zxs = []
    
    trainMinutes = int(maxT*0.5)
    testMinutes = int(trainMinutes*1.5)
    
    trainT = [y[0] for y in y if y[0] <= trainMinutes]
    trainY = [y[1] for y in y if y[0] <= trainMinutes]
    n = len(trainY)
    
    
    resY = []
    resX = []
    print('Polynomial fitting output y...')
    py = np.poly1d(np.polyfit(trainT,trainY,50))
    #plt.scatter(trainT,trainY, s=0.1, c='green', alpha=0.5)
    #plt.plot([t for t in range(0,maxT,10)], [py(t) for t in range(0,maxT,10)], linewidth=2, color='blue')
    #plt.axvline(x=trainMinutes)
    #plt.axvline(x=testMinutes)
    #plt.show()
    print('Polynomial fitting input Xs...')
    
    pxs = []
    for x in X:
        px = np.poly1d(np.polyfit([x[0] for x in x if x[0] <= trainMinutes],[x[1] for x in x if x[0] <= trainMinutes],50))
        pxs.append(px)
        plt.scatter([x[0] for x in x],[x[1] for x in x], color='green', s=0.5)
        #plt.plot([t for t in range(0,maxT,10)], [px(t) for t in range(0,maxT,10)], linewidth=2, color='blue')
        plt.axvline(x=trainMinutes)
        plt.axvline(x=testMinutes)
        plt.show()
    
    
    print('maxT',maxT)

    for t in range(0,maxT):
        zx = [px(t) for px in pxs]
        zxs.append(zx)
    
    zy = [py(t) for t in range(0,maxT)]
    
    plt.scatter([y[0] for y in y],[y[1] for y in y], s=2,alpha=0.5, color='green')
    #plt.plot([t for t in range(0,maxT, step)], zy, linewidth=2, color='black')
    print('Training model...')
    reg = BayesianRidge().fit(np.array(zxs)[:trainMinutes], zy[:trainMinutes])#MLPRegressor(hidden_layer_sizes=(2,))
    
    ##########################
    #TESTING
    #redo polyfit for test data
    py = np.poly1d(np.polyfit([y[0] for y in y],[y[1] for y in y],50))
    #plt.scatter(trainT,trainY, s=0.1, c='green', alpha=0.5)
    #plt.plot([t for t in range(0,maxT,10)], [py(t) for t in range(0,maxT,10)], linewidth=2, color='blue')
    #plt.show()
    #print('Polynomial fitting input Xs...')
    
    pxs = []
    for x in X:
        px = np.poly1d(np.polyfit([x[0] for x in x],[x[1] for x in x],50))
        pxs.append(px)
        #plt.scatter([x[0] for x in x],[x[1] for x in x], color='green', s=0.5)
        #plt.plot([t for t in range(0,maxT,10)], [px(t) for t in range(0,maxT,10)], linewidth=2, color='blue')
        #plt.show()
    
    
    #print('maxT',maxT)

    for t in range(0,maxT):
        zx = [px(t) for px in pxs]
        zxs.append(zx)
    
    zy = [py(t) for t in range(0,maxT)]
    xs = [[px(t)  for px in pxs] for t in range(0,maxT)]
    print('Predicting output...')
    ys = reg.predict(xs)
    plt.plot([t for t in range(0,maxT)], ys, color='yellow')
    pY = np.poly1d(np.polyfit([y[0] for y in y],[y[1] for y in y],50))
    plt.plot([t for t in range(0,maxT)],[pY(t) for t in range(0,maxT)], linewidth=1, c='black')
    #plt.show()
    plt.axvline(x=trainMinutes)
    plt.axvline(x=testMinutes)
    plt.show()
    # pY = np.poly1d(np.polyfit(ys[:1000],zy[:1000],3))
    # zY = [pY(y) for y in ys]
    # plt.plot([t for t in range(0,maxT, 100)], zY, linewidth=1, color='yellow')
    
    return py, pxs

if __name__ == '__main__':
    X, y = generateTrainingData()
    py, pxs = train(X,y)
