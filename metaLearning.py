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
from sklearn.ensemble import RandomForestRegressor
import warnings
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore') 
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
        typical = np.random.uniform(0.01, 10000)
        rnge = [np.random.uniform(0, typical), np.random.uniform(typical, typical*5)]
        timeSeries = []       
        freq = freqs[i]
        times = [j*freq for j in range(int(length/freq))]
        
        if random.random() > 0.99:
            for t in range(length):
                x = np.abs(np.round(np.clip(np.random.normal(typical, typical*0.1/sigma), min(rnge), max(rnge)),2))
                d = (t, x)
                timeSeries.append(d)
        else:
            
            periodNum = random.choice(range(1,4))
            amps = np.random.uniform(typical*0.01, typical*0.1, periodNum)
            periods = []
            phases = []
            for i in range(periodNum):
                period = np.abs(np.random.normal(length/2, length))/ (random.choice([3,4,5,6,7,8]))**i
                phase = length*1*random.random() 
                periods.append(period)
                phases.append(phase)
                
            for t in range(length):
                x = np.random.normal(typical, typical*0.005/sigma)  #typical distribution with noise
                for i, period in enumerate(periods):
                    wavelet = np.sin(2*np.pi*t/periods[i] + phases[i])
                    amplitude = amps[i]/(random.choice([3,4,5,6,7,8]))**i
                    x += amplitude*wavelet
                    x += np.random.normal(amplitude, np.abs(amplitude/random.choice([3,4,5,6,7,8])))
                    
                x = np.round(np.clip(x, min(rnge), max(rnge)),2)
                
                d = (t, x)
                timeSeries.append(d)
        X.append(timeSeries)
    
    inXs = []
    for i, t in enumerate(range(length)):
        xs = [x[i][1] for x in X]
        inXs.append(xs)
    
    print('Normalizing inputs and output...')
    
    miu = np.random.uniform(0, 10000)
    sigma = np.random.normal(miu, miu/random.choice([3,7,10,20, 50, 100]))
    
    mius = np.mean(inXs, axis=0)
    sigmas = np.std(inXs, axis=0) + mius/10
    
    #print(mius, sigmas)
    
    print('Randomizing parameters for underlying relationship...')
    params1 = np.random.uniform(-1,1,len(X))
    params2 = np.random.uniform(-1,1,len(X))
    params3 = np.random.uniform(-1,1,2)
    
    params4 = np.random.uniform(-1,1,2)
    params5 = np.random.uniform(-1,1,2)
    params6 = np.random.uniform(-1,1,2)
    
    randState = random.random()
    zs = []
    print('Generating underlying relationship...')
    for dx in inXs:
        z1 = np.tanh(np.dot(params1, (dx-mius)/sigmas))
        z2 = np.tanh(np.dot(params2, (dx-mius)/sigmas))
        z3 = np.array([z1,z2])
        #z = np.tanh(np.dot(params3, z3))
        
        if randState > 0.5:
            #add another layer
            z4 = np.tanh(np.dot(params4, z3))
            z5 = np.tanh(np.dot(params5, z3))
            z6 = np.array([z4,z5])
            z = np.tanh(np.dot(params6, z6))
        else:
            z = np.tanh(np.dot(params3, z3))
        zs.append(z)
    zs = np.array(zs)
    #print('z',zs)
    #print(sigma)
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

def polyfitCV(t, x):
    #n = int(trainPortion*len(t))
    #trT, teT = t[:n], t[n:]
    #trX, teX = x[:n], x[n:]
    k=10

    errors = []
    ps = []
    orders = list(range(2,50))
    for order in orders:
        err = []
        for i in range(k):
            step = int(len(t)/k)
            trT = t[:i*step] + t[(i+1)*step:]
            teT = t[i*step: (i+1)*step]
            trX = x[:i*step] + x[(i+1)*step:]
            teX = x[i*step: (i+1)*step]

            p = np.poly1d(np.polyfit(trT,trX,order))
            ps.append(p)
            predX = [p(t) for t in teT]
            error = np.mean(np.abs(np.array(predX) - np.array(teX)))
            err.append(error)
        errors.append(np.mean(err))

    minE = min(errors)
    ind = errors.index(minE)
    order = orders[ind]
    print('order', order)
    p = np.poly1d(np.polyfit(t,x,order))
    return p

def polyfit(t, x):
    #n = int(trainPortion*len(t))
    #trT, teT = t[:n], t[n:]
    #trX, teX = x[:n], x[n:]

    #plt.scatter(t,x, s=0.1, alpha=0.5)
    #plt.show()
    p = np.poly1d(np.polyfit(t,x,50))
    
    return p

def randomForestRegress(x):
    #x is a time series: x = [(t,x_t)]

    xs = [x[1] for x in x]
    ts = [[x[0]] for x in x]
    regr = RandomForestRegressor(n_estimators=50).fit(ts, xs)
    def p(t):
        t = np.array(t).reshape(-1,1)
        result = regr.predict(t)
        return result
            
    return p

def randomForestFit(t,x):
    x = list(zip(t,x))
    p = randomForestRegress(x)
    return p

def regress(t,x):
    p = randomForestFit(t,x)
    #p = polyfit(t,x)
    # xpred = [p(t) for t in range(0,max(t))]
    # iqr = np.percentile(xpred,75) - np.percentile(xpred, 25)
    # def P(xin, iqr=iqr):

    #     xout = np.median(xin) + 3*iqr*np.tanh((xin-np.median(xin))/(3*iqr))
    #     return xout
    # t = np.array(t).reshape(-1,1)
    # regr = RandomForestRegressor(n_estimators=100, n_jobs=100).fit(t,x)
    # p = lambda t: regr.predict([[t]])[0]
    
    
    return p
    ...
    
    
def train(X, y, featureImportance=True, train=0.5, test=0.25):
    print("Pulling inputs and output data from time series...")
    #maxT = max([x[-1][0] for x in X])
    
    print('Synchronizing multiple time series...')
    T = [] 
    for x in X:
        T += [d[0] for d in x]
    
    T += [d[0] for d in y]
    #T = sorted(list(set(T))) #common timestamp in X
    #print(max(T))
    maxT = max(T)
    #train-test config
    zxs = []
    
    trainLength = int(maxT*train)
    testLength = int(trainLength*((train+test)/train))
    
    trainT = [y[0] for y in y if y[0] <= trainLength]
    trainY = [y[1] for y in y if y[0] <= trainLength]
    n = len(trainY)
    
    
    resY = []
    resX = []
    print('Regressing output y...')
    py = regress(trainT,trainY)
    #plt.scatter(trainT,trainY, s=0.1, c='green', alpha=0.5)
    #plt.plot([t for t in range(0,maxT,10)], [py(t) for t in range(0,maxT,10)], linewidth=2, color='blue')
    #plt.axvline(x=trainLength)
    #plt.axvline(x=testLength)
    #plt.show()
    print('Regressing input Xs...')
    
    pxs = []
    for x in X:
        px = regress([x[0] for x in x if x[0] <= trainLength],[x[1] for x in x if x[0] <= trainLength])
        pxs.append(px)
        plt.scatter([x[0] for x in x],[x[1] for x in x], color='blue', s=0.5)
        #plt.plot([x[0] for x in x], [px(x[0])  for x in x], linewidth=2, color='blue')
        plt.axvline(x=trainLength)
        plt.axvline(x=testLength)
        plt.show()
    
    
    #print('maxT',maxT)
    
    zxs = np.array([px(list(range(0,maxT))) for px in pxs]).T
    
    # for t in range(0,maxT):
    #     zx = [px(t) for px in pxs]
    #     zxs.append(zx)
    
    zy = py(range(0,maxT))
    
    plt.scatter([y[0] for y in y],[y[1] for y in y], s=0.5,alpha=0.7, color='green')
    #plt.plot([t for t in range(0,maxT, step)], zy, linewidth=2, color='black')
    print('Training model...')
    #reg = MLPRegressor(hidden_layer_sizes=(100,), activation='tanh', max_iter=100000, solver='sgd').fit(np.array(zxs)[:trainLength], zy[:trainLength])#MLPRegressor(hidden_layer_sizes=(2,))
    if featureImportance:
        print('Analyzing Gini Feature Importances...')
        regr = RandomForestRegressor(n_estimators=100).fit(np.array(zxs)[:trainLength], zy[:trainLength])
        fi = regr.feature_importances_
    print('Analyzing underlying relationships between inputs and outputs...')
    regr = BayesianRidge(normalize=True).fit(np.array(zxs)[:trainLength], zy[:trainLength])#MLPRegressor(hidden_layer_sizes=(2,)) #RandomForestRegressor(n_estimators=100)
    #P = regress(np.array(zxs)[:trainLength], zy[:trainLength])
    ##########################
    #TESTING
    #redo polyfit for test data
    print('Testing...')
    py = regress([y[0] for y in y],[y[1] for y in y])
    #plt.scatter(trainT,trainY, s=0.1, c='green', alpha=0.5)
    #plt.plot([t for t in range(0,maxT,10)], [py(t) for t in range(0,maxT,10)], linewidth=2, color='blue')
    #plt.show()
    #print('Polynomial fitting input Xs...')
    
    pxs = []
    
    for x in X:
        #Overfitting
        px = regress([x[0] for x in x],[x[1] for x in x])
        pxs.append(px)
        #plt.scatter([x[0] for x in x],[x[1] for x in x], color='green', s=0.5)
        #plt.plot([t for t in range(0,maxT,10)], [px(t) for t in range(0,maxT,10)], linewidth=2, color='blue')
        #plt.show()
        
    #print('maxT',maxT)


    zxs = np.array([px(list(range(0,maxT))) for px in pxs]).T
    # zy = py(range(0, maxT))
    # iqr = np.percentile(zy,75) - np.percentile(zy, 25)
    # zy = np.median(zy) + 2*iqr*np.tanh((zy-np.median(zy))/(2*iqr))
    #plt.scatter(T,zy, c='black', alpha=0.5, s=0.1)
    T = [t for t in range(0,maxT)]
    xs = np.array([px(range(0,maxT)) for px in pxs]).T
    #xs = [[px(t) for px in pxs] for t in range(0,maxT)]
    print('Predicting output...')
    ys = regr.predict(xs)
      
    #iqr = np.percentile(ys,75) - np.percentile(ys, 25)

    #ys = np.median(ys) + 2*iqr*np.tanh((ys-np.median(ys))/(2*iqr))
    
    #ys = [p(x) for x in xs]
    plt.scatter(T, ys, color='blue', alpha=0.3, s=0.1)
    
    
    plt.axvline(x=trainLength)
    plt.axvline(x=testLength)
    plt.show()
    # pY = np.poly1d(np.polyfit(ys[:1000],zy[:1000],3))
    # zY = [pY(y) for y in ys]
    # plt.plot([t for t in range(0,maxT, 100)], zY, linewidth=1, color='yellow')
    
    return regr, fi

def fitPattern(xseries):
    T = [dt[0] for dt in xseries]
    #func = linear trend  + three sinusoidal waves
    def func(t, m, b, A1, A2, A3, B1,B2,B3, C1,C2,C3):
        
        A = np.array([A1,A2,A3])
        B = np.array([B1,B2,B3])
        C = np.array([C1,C2,C3])
        # print('A',A)
        # print('B',B)
        # print('C',C)
        # print('t',t)
        #  m*t + b + 
        return m*t+b + np.sum([A[i]*np.sin((2*np.pi/B[i])*t+C[i]) for i in range(len(A))], axis=0)
    
    T = [dt[0] for dt in xseries]
    X = [dt[1] for dt in xseries]
    s = np.std(X)*3
    xm = np.mean(X)
    popt, pcov = curve_fit(func, T, X, maxfev=24000, bounds=(0,[s*2/max(T), xm] + [s]*3 + [max(T)*2]*3 + [7]*3 ))
    
    # t can be a number, or an array
    f = lambda t: func(t, *popt)
    
    return f
    

if __name__ == '__main__':
    X, y = generateTrainingData()
    regr, fi = train(X,y)
    train = 0.5
    x = X[0]
    T = max([x[0] for x in X[0]])
    print('train T', int(T*train))
    p = fitPattern([v for v in x if v[0] <int(T*train)])
    
    Ts = range(T)
    plt.scatter([x[0] for x in X[0]], [x[1] for x in X[0]], s=0.5, alpha=1, color='green')
    plt.scatter(Ts, [p(t) for t in Ts], s=0.5, alpha=0.5)
    plt.axvline(x=T*train)
    plt.show()
    