# -*- coding: utf-8 -*-
from pathlib import Path
import random
import numpy as np
import json
import os, sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge, LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LinearRegression
import warnings
from scipy.optimize import curve_fit
from multiprocessing import Pool
import pickle 
import requests

warnings.filterwarnings('ignore') 
path = Path('./')

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

#randomness goes from 0 to infinity
def generateTrainingData(nx=3, length=None):
    #nx=3
    if not length:
        length = int(np.random.uniform(30, 40000))
    freqs = [int(v) for v in np.clip(np.abs(np.random.normal(10, np.random.uniform(10, 100), size=nx)), 5, length)]
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
    freq = int(np.clip(np.abs(np.random.normal(10, np.random.uniform(10, 100))), 5, length))
    times = [j*freq for j in range(int(length/freq))]
    print('generating output:','freq',freq, 'length',max(times))
    ys = []
    for t, y in outY:
        if t in times:
            ys.append([t,y])
    
    return ys, outX

def generateMultiData(basenx=10, maxRel=20, length=30000, save=True):
    iterations = random.choice(range(1, maxRel))
    print('Relationships count', iterations)
    timeSeries = []
    for iteration in range(iterations):
        print('Generating relationship', iteration+1, '/', iterations)
        y, X = generateTrainingData(nx=random.choice(range(1,basenx)), length=length)
        timeSeries.append(y)
        timeSeries += X
        
    random.shuffle(timeSeries)
    
    
    
    if save:
        print("Saving generated data...")
        with open(path / 'data' / 'fake-data' / 'randTimeSeries.json', 'w') as f:
            f.write(json.dumps(timeSeries))
    return timeSeries

def readData():
    print('Reading data...')
    with open(path / 'data' / 'fake-data' / 'randTimeSeries.json', 'r') as f:
        data = json.loads(f.read())
    return data

def ts2d(timeSeries):
    print('Converting timeSeries to dictionary...')
    fids = range(len(timeSeries))
    d = [{'fid': fid, 'series': timeSeries[fid]} for fid in fids]
    return d

def d2ts(d):
    print('Converting dictionary to timeSeries...')
    timeSeries = []
    fids = []
    for row in d:
        timeSeries.append(row['series'])
        fids.append(row['fid'])
    return timeSeries, fids



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
    #print(x[-5:])
    xs = [x[1] for x in x]
    ts = [[x[0]] for x in x]
    regr = RandomForestRegressor(n_estimators=10, max_features=0.1).fit(ts, xs)
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
    
    n = len(x)
    
    k = min(100,n)
    indices = random.sample(range(n), k)
    
    t = [v for ind, v in enumerate(t) if ind in indices]
    x = [v for ind, v in enumerate(x) if ind in indices]
    
    
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
    
    
def train(y, X, train=0.5, test=0.25, fast=False, testing=True, plot=False):
    
    if fast:
        stepRatio = 5
    else:
        stepRatio = 1
    print("Pulling inputs and output data from time series...")
    #maxT = max([x[-1][0] for x in X])
    
    print('Synchronizing multiple time series...')
    T = [] 
    for x in X:
        T += [d[0] for d in x]
    
    T += [d[0] for d in y]
    #T = sorted(list(set(T))) #common timestamp in X
    #print(max(T))
    
    
    maxT = int(max(T))
    minT = int(min(T))
    #train-test config
    zxs = []
    #print('maxT', maxT)
    trainLength = int((maxT-minT)*train + minT)
    #testLength = int(trainLength*((train+test)/train))
    
    #print('tL',trainLength)
    trainT = [y[0] for i, y in enumerate(y) if y[0] <= trainLength and i % stepRatio == 0]
    trainY = [y[1] for i, y in enumerate(y) if y[0] <= trainLength and i % stepRatio == 0]
    
    # n = len(trainY)
    
    
    # resY = []
    # resX = []
    print('Regressing output y...')
    py = regress(trainT,trainY)
    #plt.scatter(trainT,trainY, s=0.1, c='green', alpha=0.5)
    #plt.plot([t for t in range(0,maxT,10)], [py(t) for t in range(0,maxT,10)], linewidth=2, color='blue')
    #plt.axvline(x=trainLength)
    #plt.axvline(x=testLength)
    #plt.show()
    print('Regressing input Xs...')
    
    pxs = []
    for i, x in enumerate(X):
        print(i) if i % 10 == 0 else None
        px = regress([x[0] for i, x in enumerate(x) if x[0] <= trainLength and i % stepRatio == 0],[x[1] for i, x in enumerate(x) if x[0] <= trainLength and i % stepRatio == 0])
        pxs.append(px)
        if plot:
            plt.scatter([x[0] for x in x],[x[1] for x in x], color='blue', s=0.5)
            #plt.plot([x[0] for x in x], [px(x[0])  for x in x], linewidth=2, color='blue')
            plt.axvline(x=trainLength)
            #plt.axvline(x=testLength)
            plt.show()
    
    
    #print('maxT',maxT)
    
    zxs = np.array([px(list(range(minT,maxT, int((maxT-minT)/100)))) for px in pxs]).T
    
    # for t in range(0,maxT):
    #     zx = [px(t) for px in pxs]
    #     zxs.append(zx)
    
    zy = py(range(minT,maxT, int((maxT-minT)/100)))
    if plot:
        plt.scatter([y[0] for y in y],[y[1] for y in y], s=0.5,alpha=0.7, color='green')
    #plt.plot([t for t in range(0,maxT, step)], zy, linewidth=2, color='black')
    print('Training model...')
    #reg = MLPRegressor(hidden_layer_sizes=(100,), activation='tanh', max_iter=100000, solver='sgd').fit(np.array(zxs)[:trainLength], zy[:trainLength])#MLPRegressor(hidden_layer_sizes=(2,))
    
    print('Analyzing Gini Feature Importances...')
    regr = RandomForestRegressor(n_estimators=10).fit(np.array(zxs), zy)
    fi = regr.feature_importances_
    print('Analyzing underlying relationships between inputs and outputs...')
    if testing:
        regr = BayesianRidge(normalize=True).fit(np.array(zxs)[:trainLength], zy[:trainLength])#MLPRegressor(hidden_layer_sizes=(2,)) #RandomForestRegressor(n_estimators=100)
    #P = regress(np.array(zxs)[:trainLength], zy[:trainLength])
    ##########################
    #TESTING
    #redo polyfit for test data
    if testing:
        print('Testing...')
    py = regress([y[0] for i, y in enumerate(y) if i % stepRatio == 0],[y[1] for i, y in enumerate(y) if i % stepRatio == 0])
    #plt.scatter(trainT,trainY, s=0.1, c='green', alpha=0.5)
    #plt.plot([t for t in range(0,maxT,10)], [py(t) for t in range(0,maxT,10)], linewidth=2, color='blue')
    #plt.show()
    #print('Polynomial fitting input Xs...')
    
    pxs = []
    
    for j, x in enumerate(X):
        print(j) if j % 10 == 0 else None
        #Overfitting
        px = regress([x[0] for i, x in enumerate(x) if i % stepRatio == 0],[x[1] for i, x in enumerate(x) if i % stepRatio == 0])
        pxs.append(px)
        #plt.scatter([x[0] for x in x],[x[1] for x in x], color='green', s=0.5)
        #plt.plot([t for t in range(0,maxT,10)], [px(t) for t in range(0,maxT,10)], linewidth=2, color='blue')
        #plt.show()
        
    #print('maxT',maxT)


    zxs = np.array([px(list(range(minT,maxT,int((maxT-minT)/100)))) for px in pxs]).T
    zy = py(range(minT,maxT,int((maxT-minT)/100)))
    
    zxs = [v for i, v in enumerate(zxs) if i % stepRatio == 0]
    zy = [v for i, v in enumerate(zy) if i % stepRatio == 0]
    # zy = py(range(0, maxT))
    # iqr = np.percentile(zy,75) - np.percentile(zy, 25)
    # zy = np.median(zy) + 2*iqr*np.tanh((zy-np.median(zy))/(2*iqr))
    #plt.scatter(T,zy, c='black', alpha=0.5, s=0.1)
    if testing:
        T = [t for t in range(minT,maxT, int((maxT-minT)/100))]
        xs = np.array([px(range(minT,maxT, int((maxT-minT)/100))) for px in pxs]).T
        #xs = [[px(t) for px in pxs] for t in range(0,maxT)]
        print('Predicting output...')
        ys = regr.predict(xs)
          
        #iqr = np.percentile(ys,75) - np.percentile(ys, 25)
    
        #ys = np.median(ys) + 2*iqr*np.tanh((ys-np.median(ys))/(2*iqr))
        
        #ys = [p(x) for x in xs]
        if plot:
            plt.scatter(T, ys, color='blue', alpha=0.3, s=0.1)
            plt.axvline(x=trainLength)
            #plt.axvline(x=testLength)
            plt.show()
    # pY = np.poly1d(np.polyfit(ys[:1000],zy[:1000],3))
    # zY = [pY(y) for y in ys]
    # plt.plot([t for t in range(0,maxT, 100)], zY, linewidth=1, color='yellow')
    print("Retraining relationships model on full dataset...")
    model = BayesianRidge(normalize=True).fit(np.array(zxs), zy)
    print("Model successfully trained...")
    return model, fi

def fftApprox(y, thresh=5):
    print("Performing Fast Fourier Transform to extract periodicities...")
    Y = np.fft.fft(y)
    f = np.fft.fftfreq(len(y))
    psd = np.abs(Y)**2/np.sum(np.abs(Y)**2)
    threshold = np.percentile(psd, 100-thresh)
    filtered = np.array([a if a > threshold else 0 for a in psd])
    #print(filtered)
    f[f==0] =  1
    
    indices = [1 if a > 0 else 0 for a in filtered]
    periods = (1/f)[filtered>0]
    periods = np.round([p for p in periods if p > 0],1)
    #print('filtered periods', periods)
    #np.put(Y, range(cutoff, len(y)), 0.0)
    # Y = np.multiply(Y,indices)
    # ifft = np.fft.ifft(Y)
    # ifft.real
    #print('psd', psd[filtered>0])
    return periods, psd[filtered>0]

def moving_average(signal, period):
    buffer = []
    for i in range(period, len(signal)):
        buffer.append(signal[i - period : i].mean())
    return buffer

def auto_regressive(signal, p, d, q, future_count = 10):
    """
    p = the order (number of time lags)
    d = degree of differencing
    q = the order of the moving-average
    """
    buffer = np.copy(signal).tolist()
    for i in range(future_count):
        ma = moving_average(np.array(buffer[-p:]), q)
        forecast = buffer[-1]
        for n in range(0, len(ma), d):
            forecast -= buffer[-1 - n] - ma[n]
        buffer.append(forecast)
    return buffer


def fitPattern(xseries, plot=False):
    
    T = [dt[0] for dt in xseries]
    X = [dt[1] for dt in xseries]
    
    n = len(xseries)
    
    k = min(100,n)
    indices = random.sample(range(n), k)
    
    T = [v for ind, v in enumerate(T) if ind in indices]
    X = [v for ind, v in enumerate(X) if ind in indices]
    
    
    train = max(int(0.5*len(T)), 1)
    periods, psd = fftApprox(X, thresh=3) 
    periods *= (max(T) - min(T))/(len(T)-1)
    l = max(T) - min(T)
    if len(periods) > 30:
        step = 3
    elif len(periods) > 120:
        step = 5
    else:
        step = 1
    periods = list(set(list(periods)[1:: 10])) + [l*2, l*3, l*4]
    #periods = periods
    print('periods',periods)
    def func(t, b, c, A, B, C):
        return b*t + c + A*np.sin((2*np.pi/B)*t) + C*np.cos((2*np.pi/B)*t)#C)# + D*np.tanh(E*t+F)
    
    errors = []
    
    for period in periods:
        indices = random.sample(range(len(T)), train)
        trT = [t for ind, t in enumerate(T) if ind in indices]
        trX = [x for ind, x in enumerate(X) if ind in indices]
        popt, pcov = curve_fit(func, trT, trX, p0=[0, np.mean(trX), np.std(trX)*2, period , np.std(trX)*2], maxfev=300000) #max(trT)/(i+1), np.max(trX), 1/max(trT), 1/max(trT)
        Xpred = func(np.array(T), *popt)
        
        error = np.sqrt(np.mean((np.array(Xpred)-np.array(X))**2))/np.mean(X) #+ 0.2*i + 0.1*popt[2]/np.mean(X) + 0*popt[3]/max(T)
        
        errors.append(error)
    
    if plot:
    
        plt.plot(np.array(errors).reshape(-1,1), linewidth=1)
        plt.show()
    
    #period = periods[np.argmax(psd)]
    
    ind = np.argmin(errors)
    popt, pcov = curve_fit(func, T, X, p0=[0, np.mean(X), np.std(X)*2, periods[ind], np.std(trX)*2], maxfev=3000000) # np.max(X), 1/max(T), 1/max(T)
        
    f = lambda t: func(t, *popt)
    p = lambda t: list(map(f,t))
    return p
    
def forecast(y, X=None, model=None, predLength=0.3, plot=False):
    
    print("Forecasting time-series...")
    pxs = []

    
    if not X:
        print("Predicting time-series without assuming any underlying relationships...")
        T = [v[0] for v in y]
        p = fitPattern(y) # <int(T*train)
        maxT = np.max(T)
        
        dT = T[-1] - T[-2]
        # if fast:
        #     dT = dT * 100
        counts = int(len(T) * (predLength))
        
        k1 = min(50, len(T))
        indices1 = random.sample(range(len(T)), k1)
        
        T = [v for i, v in enumerate(T) if i in indices1] + [T[-1] + (count+1)*dT for i, count in enumerate(range(counts)) if i % 10 == 0]
        #T = [T[-1], [T[-1] + (count+1)*dT for count in range(counts)][-1]]
        ypred = np.clip(p(T), 0, np.inf)
        noise = np.std(np.array([v[1] for i, v in enumerate(y) if i in indices1]) - ypred[:k1])
        bandwidth = noise*3
        #print("bandwidth", bandwidth)
        Upp = np.clip(ypred + bandwidth, 0, np.inf)
        Lwr = np.clip(ypred - bandwidth, 0, np.inf)
        
        
        
        if plot:
            plt.scatter([v[0] for v in y], [v[1] for v in y], s=0.5, alpha=1, color='green')
            plt.scatter(T, ypred, s=1, alpha=1, color='black')
            plt.scatter(T, Upp, s=0.5, alpha=1, color='orange')
            plt.scatter(T, Lwr, s=0.5, alpha=1, color='red')
            plt.axvline(x=maxT)
            plt.show()
    
    else:
        print('Dependent inputs time-series detected, using inputs to predict output...')
        for i, x in enumerate(X):
            #print('x', x)
            
            # print('train T', int(T*train))
            print("Analyzing individual input time-series...")
            p = fitPattern(x) # <int(T*train)
            
            pxs.append(p)
        
        T = [v[0] for v in y]
        #print(T)
        maxT = np.max(T)
        dT = int((T[-1] - T[0])/(len(T)-1))
        # if fast:
        #     dT = dT * 100
        #     step = int(len(T)/100)+1
        # else:
        #     step = 1
        counts = int(len(T) * (predLength))
    
        k1 = min(50, len(T))
        indices1 = random.sample(range(len(T)), k1)
        
        T = [v for i, v in enumerate(T) if i in indices1] + [T[-1] + (count+1)*dT for i, count in enumerate(range(counts)) if i % 10 == 0]
        
        #T = T + [T[-1] + (count+1)*dT for count in range(counts)]
        #T = [T[-1], [T[-1] + (count+1)*dT for count in range(counts)][-1]]
        #print(T)
        PX = np.array([p(T) for p in pxs]).T
        ypred1 = model.predict(PX)
        
        T = [v[0] for v in y]
        p = fitPattern(y) # <int(T*train)
        maxT = np.max(T)
        
        dT = int((T[-1] - T[0])/(len(T)-1))
        # if fast:
        #     dT = dT * 100
        #     step = int(len(T)/100) + 1
        # else:
        #     step = 1
        counts = int(len(T) * (predLength))
        
        # k1 = min(100, len(T))
        # indices1 = random.sample(range(len(T)), k1)
        
        T = [v for i, v in enumerate(T) if i in indices1] + [T[-1] + (count+1)*dT for i, count in enumerate(range(counts)) if i % 10 == 0]
        #T = T + [T[-1] + (count+1)*dT for count in range(counts)]
        #T = [T[-1], [T[-1] + (count+1)*dT for count in range(counts)][-1]]
        ypred2 = p(T)
        
        #ypred = (ypred1 + ypred2)/2
        c = np.array([ypred1, ypred2])
        ymax = np.max(c, axis=0)
        ymin = np.min(c, axis=0)
        ypred = np.clip(np.mean(c, axis=0), 0, np.inf)

        noise = np.std(np.array([v[1] for i, v in enumerate(y) if i in indices1]) - ypred[:k1])
        bandwidth = noise*6
        #print("bandwidth", bandwidth)
        Upp = np.clip(np.mean(np.array([ypred + bandwidth, ymax]), axis=0), 0, np.inf)
        Lwr = np.clip(np.mean(np.array([ypred - bandwidth, ymin]), axis=0), 0, np.inf)
        # Upp = ypred + bandwidth
        # Lwr = np.abs(ypred - bandwidth)
        
        if plot:
            plt.scatter([v[0] for v in y], [v[1] for v in y], s=0.5, alpha=1, color='green')
            # plt.scatter(T, ypred, s=0.5, alpha=1, color='blue')
            # plt.scatter(T, Upp, s=0.5, alpha=1, color='orange')
            # plt.scatter(T, Lwr, s=0.5, alpha=1, color='red')
            plt.plot(T, ypred, linewidth=1, color='blue')
            plt.plot(T, Upp, linewidth=1, color='orange')
            plt.plot(T, Lwr, linewidth=1, color='red')
            plt.axvline(x=maxT)
            plt.show()
    
    yout = list(zip(T, ypred))
    yupp = list(zip(T, Upp))
    ylwr = list(zip(T, Lwr))
    print(ylwr[-1][1])
    print(y[-1][1])
    print(yupp[-1][1])
    print(indices1[-1])
    
    anomalies = []
    vals = []
    for i,v in enumerate(y):
        if i in indices1:
            vals.append(v[1])
    
    for i, v in enumerate(vals):
        if not (ylwr[i][1] <= v <= yupp[i][1]):
            anomalies.append(v)
    
    lwr = ylwr[k1][1]
    upp = yupp[k1][1]
    isAnomaly = not lwr <= y[-1][1] <= upp
    
    
    
    
    # anomalies = [v for i, v in enumerate(y) if not (ylwr[i][1] <= v[1] <= yupp[i][1]) and i in indices1]
    anomalyRate = len(anomalies)/len([y for i, y in enumerate(y) if i in indices1])
    print('anomalies', anomalies)
    print('anomalyRate', anomalyRate)
    
    ydict = {'pred':yout, 'high': yupp, 'low': ylwr, 'anomalyRate': anomalyRate, 'anomalies': anomalies, 'isAnomaly': isAnomaly, 'highNow': upp, 'lowNow': lwr, 'yNow': y[-1][1]}
    
    return ydict

def controlStrategy(y,X, model, control=[0], maximize=True, predLength=0.3, ydict=None):
    #X is a list of dependent time-series, y is the target time-series.
    #control is a list of control variables (indices for X)
    #mode: min or max
    #regr is the model y = regr.predict(X)
    
    #learning the constraints (control boundaries)
    
    if not control:
        print('No control variables defined, nothing to control...')
        return None, None
    elif control == [-1]:
        control = list(range(len(X)))
    
    print('Figuring out feasible region for output variable...')
    if not ydict:
        ydict = forecast(y,X, model, predLength=predLength)
    tmax = y[-1][0]
    print('tmax', tmax)
    print('Figuring out feasible regions for input variables...')
    Xdict = [forecast(x, predLength=predLength) for x in X]
    #print([[v[1] for v in x['low']] for x in Xdict])
    # Xlow = [np.percentile([v[1] for v in x['pred'][:int(tmax*(1-predLength))]], 25) for x in Xdict]
    # Xhigh = [np.percentile([v[1] for v in x['high'][:int(tmax*(1-predLength))]], 75) for x in Xdict]
    
    Xlow = [np.percentile([v[1] for v in x], 5) for x in X]
    Xhigh = [np.percentile([v[1] for v in x], 95) for x in X]

    tpred = ydict['pred'][-1][0]
    print('Prediction tpred',tpred)
    
    #synchronizing xinput on tpred
    print('Synchronizing inputs on tpred..')
    xin = [x['pred'][-1000:] for x in Xdict]
    print([x['pred'][-1] for x in Xdict])
    
    pxs = [regress([v[0] for v in x], [v[1] for v in x]) for x in xin] #fitPattern(x)
    
    xbase = np.array([px([tpred])[0] for px in pxs])
    
    print('Using control variables to define Evolutionary Strategy variance...')
    variance = xbase/10
    #print(variance)
    for i, var in enumerate(variance):
        print('control', control)
        if i not in control:
            variance[i] = 0
            Xlow[i] = 0
            Xhigh[i] = np.inf
    print('Xlow', Xlow)
    print('Xhigh', Xhigh)
    
    if maximize:
        optimize = np.argmax
    else:
        optimize = np.argmin
    
    xopt = xbase
    print('Using Evolutionary Strategy to find optimal control...')
    for iteration in range(10):
        print('iteration', iteration)
        xin = np.array([np.random.normal(xopt, variance) for i in range(30)] + [xopt])
        #print('xinput',xin)
        xin = [np.clip(x, Xlow, Xhigh) for x in xin]

        yout = np.clip(model.predict(xin), 0, np.inf)
        #print('yout',yout)
        
        tolerance = np.std(yout)/np.mean(yout)
        print('tolerance', tolerance)
        ind = optimize(yout)
        xopt1 = xin[ind]
        dx = np.mean(np.abs(np.array(xopt) - np.array(xopt1)))/np.mean(xopt)
        yopt = yout[ind]
        xopt = xopt1
        print('dx', dx)
        variance *= dx*10*(iteration+1)
        print('variance', variance)
        if tolerance < 0.0001:
            print('Best solution found by convergence...')
            break
        
    print('control', control)
    tlast = y[-1][0]
    xlast = [x[-1][1] for x in X]
    ylast = y[-1][1]
    print('tlast', tlast)
    print('Xlast', xlast)
    print('Ylast', ylast)
    print('\n')
    
    ybase = np.clip(model.predict([xbase])[0],0, np.inf)
    print('tbase', tpred)
    print('xbase', xbase)
    print('ybase', ybase)
    print('\n')
    
    topt = tpred
    print('topt', tpred)
    print('xopt', xopt)
    print('yopt', yopt)
    
    result = {'last':{'t': tlast, 'X':xlast, 'y':ylast},
              'base':{'t':tpred, 'X':list(xbase), 'y':ybase},
              'opt':{'t':tpred, 'X': list(xopt), 'y':yopt}}
    
    return result
    ...

def findRelationships(X):
    #find all possible variable-interdependencies in timeSeries data X.
    #output should be a [fids X fids] matrix of feature importances along with relationship models.
    t1 = time.time()
    models = []
    fis = []
    
    global findRel
    
    def findRel(x):
        i = X.index(x)
        print('Analyzing feature', i, '/', len(X)-1)
        model, fi = train(x, X[:i] + X[i+1:], fast=True, testing=False)
        fi = list(fi)
        fi.insert(i, -1)
        return model, fi
    
    with Pool(7) as p:
        result = p.map(findRel, X)
    # for i, x in enumerate(X):
    #     print('Analyzing feature', i, '/', len(X)-1)
    #     model, fi = train(x, X[:i] + X[i+1:], fast=True)
    #     list(fi).insert(i, 0)
    #     fis.append(fi)
    #     models.append(model)
    t2 = time.time() - t1
    print('time taken',t2/60)
    
    print('Saving model...')
    with open(path / 'models'/ 'relationship.model', 'wb') as f:
        pickle.dump(result, f)

    return result

def requestJs():
    print('Sending requests...')
    response = requests.get('http://192.168.101.21:18888/adapter/datastream?start=1451904960&end=1451975040')
    js = response.json()
    time = js['tick']
    mapping = js['mapping']
    data = js['data']
    
    response = requests.get('http://192.168.101.21:18888/adapter/oplog')
    controlDecision = response.json()
    
    d = {'time': time, 'mapping': mapping, 'data': data, 'controlDecision': controlDecision}
    
    print('Saving response data...')
    with open(path / 'data' / 'fake-data' / 'requested.json', 'w') as f:
        f.write(json.dumps(d))
    
    return time, mapping, controlDecision, data
    
def loadData():
    with open(path / 'data' / 'fake-data' / 'requested.json', 'r') as f:
        d = json.loads(f.read())
    
    mapping = d['mapping']
    controlDecision = d['controlDecision']
    data = d['data']
    
    fids = [row['id'] for row in data]
    X = [row['series'] for row in data]
    ts = []
    Fids = []
    for i, x in enumerate(X):
        if x:
            series = []
            for row in x:
                if row != [None,None]:
                    if row[1] >= 0:
                        series.append(row)
                else:
                    print('NONE NONE')
            #print(series[-5:])
            ts.append(series)
            Fids.append(fids[i])
    
    
    return ts, Fids, mapping, controlDecision


def analysis(fis, forecasts, fids, controlStrategies=None):
    
    anomalyIndices = []
    forecasts = forecasts['forecast']
    for ind, forecast in enumerate(forecasts):
        print('forecast',forecast)
        if random.random() > 0.5: #forecast['isAnomaly']
            anomalyIndices.append(ind)
    
    anomalyFids = [fids[ind] for ind in anomalyIndices]
    
    summary = []
    if anomalyFids:
        for ind, anomalyFid in enumerate(anomalyFids):
            sentence = '指标' + str(anomalyFid) + '出现异常：当前值' + str(round(forecasts[ind]['yNow'],2)) + '不在正常范围内' + ' (' + str(round(forecasts[ind]['lowNow'],2)) + '~' + str(round(forecasts[ind]['highNow'],2))  + ') '
            summary.append(sentence)
        
    if not summary:
        summary.append('无异常')
    
    return summary



def loadModel():
    print('Loading model...')
    with open(path / 'models'/ 'relationship.model', 'rb') as f:
        model = pickle.load(f)

    return model

def featureImportances(models, fids):
    d = {'datetime': time.time(), 'featureImportances': []}
    for i, fid in enumerate(fids):
        d['featureImportances'].append({'fid':fid, 'featureImportance': list(zip(fids, models[i][1]))})
        
    return d

def multiForecast(X, models, fids, predLength=0.3):
    
    global forecastParallel
    
    def forecastParallel(y):
        
        i = X.index(y)
        print('Forecasting feature', i, '/', len(X)-1)
        ydict = forecast(y, X[:i] + X[i+1:], models[i][0], predLength=predLength, plot=False)
        return ydict
    
    with Pool(7) as p:
        ydicts = p.map(forecastParallel, X)
        
    result = {'datetime':time.time(), 'forecast': []}
    
    for i, ydict in enumerate(ydicts):
        fid = fids[i]
        result['forecast'].append({'fid': fid, **ydict})
    
    return result
    #ydict = forecast(y, X=None, model=None, predLength=predLength, fast=True, plot=False)

def controlStrategiesRandom(X, models, fids, predLength=0.3):
    
    y = random.choice(X)
    targetIndex = X.index(y)
    maximization = round(random.random())
    mode = 'max' if maximization else 'min'
    
    model = models[targetIndex][0]
    
    controlVars = random.sample(fids[:targetIndex]+fids[targetIndex+1:], k=random.choice(range(1,4)))
    print('controlVars', controlVars)
    
    controlVarsIndices = []
    newfids = fids[:targetIndex]+fids[targetIndex+1:]
    for cv in controlVars:
        controlVarsIndices.append(newfids.index(cv))
    
    
    cs = controlStrategy(y,X[:targetIndex] + X[targetIndex+1:], model, control=controlVarsIndices, maximize=maximization, predLength=predLength)
    
    cs['last']['X'].insert(targetIndex, cs['last']['y'])
    cs['base']['X'].insert(targetIndex, cs['base']['y'])
    cs['opt']['X'].insert(targetIndex, cs['opt']['y'])
    
    
    result = {'datetime': time.time(), 'mode': mode, 'targetFid': fids[targetIndex], 'controls': controlVars, 'fidsList': fids, **cs}
    return result
    
def generateJson(rows=100, columns=5, outputFilename=None):
    requestJs(n=rows)
    #X = loadData()
    ts, fids, mapping, controlDecision = loadData()
    #X = readData()[:columns]
    # d = ts2d(X)
    # ts, fids = d2ts(d)
    result = findRelationships(ts)
    models = loadModel()
    fis = featureImportances(models, fids)
    forecasts = multiForecast(X, models, fids, predLength=0.1)
    controlStrategies = controlStrategiesRandom(X, models, fids)
    sentences = analysis(fis, forecasts, fids)
    print('summary',sentences)
    jsdict = {'datetime': time.time(), 'featureImportances': fis, 'forecasts': forecasts, 'controlStrategies': controlStrategies, 'summary': sentences}
    if not outputFilename:
        outputFilename = 'output.json'
    with open(path / 'data' / 'fake-data' / outputFilename, 'w') as f:
        f.write(json.dumps(jsdict))
    print('JSON done.')
    
def knnRegress(X, n_points=30):
    #for downsampling, denoising and synchronizing signals
    T = []
    for x in X:
        T += [r[0] for r in x]
      
    T = list(set(T))
    minT = int(min(T))
    maxT = int(max(T))
    
    Ts = range(minT, maxT+1*int((maxT-minT)/n_points), int((maxT-minT)/n_points))
    
    trX = []
    for t in Ts:
        trx = []
        for x in X:
            ts = (1/(np.abs((np.array([r[0] for r in x if r[0]]) -  t)) + 0.1))**1 # <= (minT + (t-minT)*1)
            ps = ts/np.sum(ts)
            #print('sum', round(np.sum(ps),2))
            xs = np.array([r[1] for r in x if r[0]]) # <= (minT + (t-minT)*1)
            rx = np.dot(xs,ps)
            trx.append(rx)
        trX.append(trx)
    
    trX = np.array(trX)
    Ts = np.array(Ts)
    
    return Ts, trX

def forecast2(Ts,trX, S=0.7, L=0.5):
    s = int(len(trX)*S)
    l = int(len(trX)*L)
    F = len(trX[0])
    #Y = trX
    #print(len(Ts))
    regrs = []
    resYs = []
    for ind in range(F):
        #print('training', ind, '/', F)
        trainX = np.array([trX[i:s+i, ind] for i in range(len(trX)-s)])
        #print(np.shape(trainX))
        trainY = np.array([trX[s+i,ind] for i in range(len(trX)-s)])
        # print(np.shape(trainX))
        #print(np.shape(trainY))
        regr = BayesianRidge().fit(trainX, trainY)
        regrs.append(regr)
        #print('training done')
        resY = []
        
        fcX = trX[-s:, ind]
        #print('forecasting')
        for j in range(l):
            #print(fcX)
            #print(np.shape(fcX))
            #print(np.array([fcX]))
            fcY = regr.predict([fcX])
            #print(fcY)
            #Y = np.append(Y, fcY[0])
            fcX =  np.append(fcX, fcY[0])
            fcX = fcX[-s:]
            #trX = trX.T
            resY.append(fcY[0])
        resYs.append(resY)
    
    YM = np.array(resYs).T
    trX = list(trX) + list(YM)
    trX = np.array(trX)
            
    for j in range(l):  
        Ts = np.append(Ts, Ts[-1] + (Ts[1]-Ts[0]))
    

    print(len(Ts))
    print(len(trX))    
    return Ts, trX
    
    
    ...

def featureImportances2(Ts, trX):
    
    models = []
    fis = []
    M = trX.T
    for i, trx in enumerate(M):
        print(i, '/', len(M))
        trainX = [[v for v in r] for r in trX.T]
        trainX = trainX[:i] + trainX[i+1:]
        trainX = np.array(trainX).T
        
        #print(np.std(trainX))
        regr = RandomForestRegressor(n_estimators=30, max_features='sqrt').fit(trainX, trX[:,i])
        #trainX[:,i] = vec
        fi = list(regr.feature_importances_)
        fi.insert(i,-1)
        models.append(regr)
        fis.append(fi)

    
    print('completed')
        #fis.append(fi)
    #fis = np.array(fis)
    return models, fis



if __name__ == '__main__':
    # y, X = generateTrainingData()
    # t0 = time.time()
    # t1 = time.time()
    # with HiddenPrints():
    #     model, fi = train(y,X, fast=True)
    # t2 = time.time()
    # print('training time taken', t2-t1)
    # # t1 = time.time()
    # # with HiddenPrints():   
    # #     #ypred, yhigh, ylow, anomalyRate = forecast(y,X=None, regr=None)
    # # t2 = time.time()
    # # print('time taken', t2-t1)
    # t1 = time.time()
    # with HiddenPrints():
    #     ydict = forecast(y,X, model, plot=True, predLength=0.2)
    
    # t2 = time.time()
    # print('forecast time taken', t2-t1)
    # t1 = time.time()
    # with HiddenPrints():
    #     strategy = controlStrategy(y, X, model=model, control=[0], maximize=True, predLength=0.2, ydict=ydict)
        
    # t2 = time.time()
    # print('strategy time taken', t2-t1)
    # print('total time', t2-t0)
    
    # sentences = analysis(fi, [ydict], fids=['mds7392'])
    # print(sentences)
    # # t1 = time.time()
    
    # model, fi = train(y,X)
    
    # #ypred, yhigh, ylow, anomalyRate = forecast(y,X=None, regr=None)
    # ydict = forecast(y,X, model, plot=True, predLength=0.1)
    
    # strategy = controlStrategy(y, X, model=model, control=[0], maximize=True, predLength=0.1)
        
    # t2 = time.time()
    
    # print('time taken', t2-t1)
    ######################
    # #X = generateMultiData(length=30000) 
    # t1 = time.time()
    # X = loadModel()
    # #with HiddenPrints():
    # generateJson(outputFilename='testOutput3.json')
    # t2 = time.time()
    # print('time taken', t2-t1)
    # # #######################
    #t1 = time.time()
    #rows = 10
    #requestJs()
    #X = loadData()
    #ts, fids, mapping, controlDecision = loadData()
    # #result = findRelationships(ts)
    # X = ts[:]
    # x = random.choice(X)
    # i = X.index(x)
    # print('Analyzing feature', i, '/', len(X)-1)
    # model, fi = train(x, X[:i] + X[i+1:], fast=True, testing=False)
    # fi = list(fi)
    # fi.insert(i, -1)
    # ydict = forecast(x,X[:i] + X[i+1:], model, plot=True, predLength=0.2)
    
    # strategy = controlStrategy(x, X[:i] + X[i+1:], model=model, control=random.sample(fids, 3), maximize=True, predLength=0.1)
    
    # t2 = time.time()
    # print(t2-t1)
    
    ######################################
    t1 = time.time()
    #X, fids, mapping, controlDecision = loadData()
    #I = [i for i,x in enumerate(X) if 2 < len(x) < 300]
    #ind = random.choice(I)
    with open(path / 'data' / 'fake-data' / 'randTimeSeries.json', 'r') as f:
        X = json.loads(f.read())
    ind = random.choice(range(len(X)))
    
    Ts, trX = knnRegress(X, n_points=30)
    
    Ts, resY = forecast2(Ts, trX)
    
    plt.scatter([r[0] for r in X[ind]], [r[1] for r in X[ind]], s=20, alpha=0.5, c='green')
    plt.scatter(Ts, resY[:,ind], s=10, c='black')
    plt.plot(Ts, resY[:,ind], linewidth=1, c='blue')
    #plt.plot()
    plt.show()
    
    #models, fis = featureImportances2(Ts, trX)
    t2 = time.time()
    print('time taken', t2-t1)
    
    
    
    