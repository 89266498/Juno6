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
    
    
def train(y, X, train=0.5, test=0.25):
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
    
    print('Analyzing Gini Feature Importances...')
    regr = RandomForestRegressor(n_estimators=100).fit(np.array(zxs), zy)
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
    zy = py(range(0,maxT))
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
    train = max(int(0.5*len(T)), 1)
    periods, psd = fftApprox(X, thresh=5) 
    periods *= (max(T) - min(T))/(len(T)-1)
    l = max(T) - min(T)
    if len(periods) > 30:
        step = 3
    elif len(periods) > 120:
        step = 5
    else:
        step = 1
    periods = list(set(list(periods)[1:: step])) + [l*2, l*3, l*4]
    #periods = periods
    print('periods',periods)
    def func(t, b, c, A, B, C):
        return b*t + c + A*np.sin((2*np.pi/B)*t + C)# + D*np.tanh(E*t+F)
     
    errors = []
    
    for period in periods:
        indices = random.sample(range(len(T)), train)
        trT = [t for ind, t in enumerate(T) if ind in indices]
        trX = [x for ind, x in enumerate(X) if ind in indices]
        popt, pcov = curve_fit(func, trT, trX, p0=[0, np.mean(trX), np.std(trX)*2, period , 1/max(trT)], maxfev=3000000) #max(trT)/(i+1), np.max(trX), 1/max(trT), 1/max(trT)
        Xpred = func(np.array(T), *popt)
        
        error = np.sqrt(np.mean((np.array(Xpred)-np.array(X))**2))/np.mean(X) #+ 0.2*i + 0.1*popt[2]/np.mean(X) + 0*popt[3]/max(T)
        
        errors.append(error)
    
    if plot:
    
        plt.plot(np.array(errors).reshape(-1,1), linewidth=1)
        plt.show()
    
    #period = periods[np.argmax(psd)]
    
    ind = np.argmin(errors)
    popt, pcov = curve_fit(func, T, X, p0=[0, np.mean(X), np.std(X)*2, periods[ind], 1/max(T),], maxfev=3000000) # np.max(X), 1/max(T), 1/max(T)
        
    f = lambda t: func(t, *popt)
    p = lambda t: list(map(f,t))
    return p
    
def forecast(y, X=None, model=None, predLength=0.3, fast=False, plot=False):
    
    print("Forecasting time-series...")
    pxs = []

    
    if not X:
        print("Predicting time-series without assuming any underlying relationships...")
        T = [v[0] for v in y]
        p = fitPattern(y) # <int(T*train)
        maxT = np.max(T)
        
        dT = T[-1] - T[-2]
        if fast:
            dT = dT * 100
        counts = int(len(T) * (predLength))
        T = T + [T[-1] + (count+1)*dT for count in range(counts)]
        ypred = np.clip(p(T), 0, np.inf)
        noise = np.std(np.array([v[1] for v in y]) - ypred[:len(y)])
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
        dT = T[-1] - T[-2]
        if fast:
            dT = dT * 100
        counts = int(len(T) * (predLength))
        T = T + [T[-1] + (count+1)*dT for count in range(counts)]
        #print(T)
        PX = np.array([p(T) for p in pxs]).T
        ypred1 = model.predict(PX)
        
        T = [v[0] for v in y]
        p = fitPattern(y) # <int(T*train)
        maxT = np.max(T)
        
        dT = T[-1] - T[-2]
        if fast:
            dT = dT * 100
        counts = int(len(T) * (predLength))
        T = T + [T[-1] + (count+1)*dT for count in range(counts)]
        ypred2 = p(T)
        
        #ypred = (ypred1 + ypred2)/2
        c = np.array([ypred1, ypred2])
        ymax = np.max(c, axis=0)
        ymin = np.min(c, axis=0)
        ypred = np.clip(np.mean(c, axis=0), 0, np.inf)
        
        noise = np.std(np.array([v[1] for v in y]) - ypred[:len(y)])
        bandwidth = noise*6
        #print("bandwidth", bandwidth)
        Upp = np.clip(np.mean(np.array([ypred + bandwidth, ymax]), axis=0), 0, np.inf)
        Lwr = np.clip(np.mean(np.array([ypred - bandwidth, ymin]), axis=0), 0, np.inf)
        # Upp = ypred + bandwidth
        # Lwr = np.abs(ypred - bandwidth)
        
        if plot:
            plt.scatter([v[0] for v in y], [v[1] for v in y], s=0.5, alpha=1, color='green')
            plt.scatter(T, ypred, s=0.5, alpha=1, color='blue')
            plt.scatter(T, Upp, s=0.5, alpha=1, color='orange')
            plt.scatter(T, Lwr, s=0.5, alpha=1, color='red')
            plt.axvline(x=maxT)
            plt.show()
    
    yout = list(zip(T, ypred))
    yupp = list(zip(T, Upp))
    ylwr = list(zip(T, Lwr))
    
    anomalies = [v for i, v in enumerate(y) if not ylwr[i][1] <= v[1] <= yupp[i][1]]
    anomalyRate = len(anomalies)/len(y)
    print('anomalies', anomalies)
    print('anomalyRate', anomalyRate)
    
    ydict = {'pred':yout, 'high': yupp, 'low': ylwr, 'anomalyRate': anomalyRate, 'anomalies': anomalies}
    
    return ydict

def controlStrategy(y,X, model, control=[0], maximize=True, predLength=0.3):
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
    for iteration in range(100):
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
              'base':{'t':tpred, 'X':xbase, 'y':ybase},
              'opt':{'t':tpred, 'X':xopt, 'y':yopt}}
    
    return result
    ...
    
if __name__ == '__main__':
    y, X = generateTrainingData()
    model, fi = train(y,X)
    
    #ypred, yhigh, ylow, anomalyRate = forecast(y,X=None, regr=None)
    ydict = forecast(y,X, model, plot=True, predLength=1)
    
    strategy = controlStrategy(y, X, model=model, control=[0], maximize=True, predLength=1)
    