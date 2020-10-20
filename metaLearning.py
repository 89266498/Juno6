# -*- coding: utf-8 -*-
from pathlib import Path
import random
import numpy as np
import json
import os, sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge, LinearRegression, RidgeCV, LassoCV
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



def loadModel():
    print('Loading model...')
    with open(path / 'models'/ 'relationship.model', 'rb') as f:
        model = pickle.load(f)

    return model


    
def knnRegress(X, n_points=30):
    #for downsampling, denoising and synchronizing signals
    T = []
    for x in X:
        T += [r[0] for r in x]
      
    T = list(set(T))
    minT = int(min(T))
    maxT = int(max(T))
    
    Ts = range(minT, maxT+0*int((maxT-minT)/n_points), int((maxT-minT)/n_points))
    
    trX = []
    for t in Ts:
        trx = []
        for x in X:
            ts = (1/(np.abs((np.array([r[0] for r in x]) -  t)) + 0.1))**3 # <= (minT + (t-minT)*1)
            ps = ts/np.sum(ts)
            #print('sum', round(np.sum(ps),2))
            xs = np.array([r[1] for r in x]) # <= (minT + (t-minT)*1)
            rx = np.dot(xs,ps)
            trx.append(rx)
        trX.append(trx)
    
    trX = np.array(trX)
    Ts = np.array(Ts)
    
    resX = np.array([list(zip(Ts, x)) for x in trX.T])
    
    #print(resX[0])
    return resX

def forecast2(trX, fids, S=0.5, L=0.5, anomalyRate=0.001):
    print("Forecasting...")
    
    Ts = np.array([r[0] for r in trX[0]])
    trX = [[r[1] for r in x] for x in trX]
    trX = np.array(trX).T
    s = int(len(trX)*S)
    l = int(len(trX)*L)
    F = len(trX[0])
    #Y = trX
    #print(len(Ts))
    regrs = []
    resYs = []
    C = []
    I = []
    upps = []
    lwrs = []
    for ind in range(F):
        #print('training', ind, '/', F)
        trainX = np.array([trX[i:s+i, ind] for i in range(len(trX)-s)])
        #print(np.shape(trainX))
        trainY = np.array([trX[s+i,ind] for i in range(len(trX)-s)])
        # print(np.shape(trainX))
        #print(np.shape(trainY))
        regr = BayesianRidge(normalize=False, fit_intercept=False).fit(trainX, trainY)  #RidgeCV(normalize=True, alphas=[1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]), LassoCV(normalize=True, fit_intercept=False, alphas=10**np.array(list(range(2,8))))

        coefs = regr.coef_

        intercept = regr.intercept_
        
        
        #print('intercept', intercept)
        
        #coefs = coefs / np.linalg.norm(coefs)
        #coefs = np.clip(coefs, -0.5,0.5) 
        #print('mag',np.linalg.norm(coefs))
        #print('coefs', coefs)
        C.append(coefs)
        I.append(intercept)
        regrs.append(regr)
        #print('training done')
        resY = []
        upp = []
        lwr = []
        fcX = trX[-s:, ind]
        
        maxX = np.percentile(trX[:,ind], (1 - anomalyRate/2)*100)
        minX = np.percentile(trX[:,ind], anomalyRate*100/2)
        
        #print('forecasting')
        for j in range(l):
            #print(fcX)
            #print(np.shape(fcX))
            #print(np.array([fcX]))
            fcY, std = regr.predict([fcX], return_std=True)#, return_std=True
            #fcY = [np.dot(coefs, fcX) + intercept]
            #print(fcY)
            #print(std)
            #Y = np.append(Y, fcY[0])
            fcX =  np.append(fcX, fcY[0])
            upp.append(list(np.clip(fcY[0] + 3*std, 0, np.inf))[0])
            lwr.append(list(np.clip(fcY[0] - 3*std, 0, np.inf))[0])
            fcX = fcX[-s:]
            #trX = trX.T
            resY.append(np.clip(fcY[0], 0 , np.inf))

        resYs.append(resY)
        upps.append(upp)
        lwrs.append(lwr)
        
    YM = np.array(resYs).T
    trX = list(trX) + list(YM)
    trX = np.array(trX)
    upps = upps
    lwrs = lwrs
    for j in range(l):  
        Ts = np.append(Ts, Ts[-1] + (Ts[1]-Ts[0]))
    
    Ts1 = Ts[-l:]

    resX = [list(zip(list(Ts), list(x))) for x in trX.T]
    upps = [list(zip(list(Ts1), list(x))) for x in upps]
    lwrs = [list(zip(list(Ts1), list(x))) for x in lwrs]
    #print(len(Ts))
    #print(len(trX))  
    ydicts = []
    for i, yout in enumerate(resX):
        isAnomaly = not (lwrs[i][0][1] <= yout[-l][1] <= upps[i][0][1])
        ydict = {'pred':yout, 'high': upps[i], 'low': lwrs[i], 'anomalyRate': anomalyRate, 'isAnomaly': isAnomaly, 'highNow': upps[i][0][1], 'lowNow': lwrs[i][0][1], 'yNow': yout[-l][1], 'tNow': Ts[-1]}
        ydicts.append(ydict)
    
    result = {'datetime':time.time(), 'forecast': []}
    for i, ydict in enumerate(ydicts):
        fid = fids[i]
        result['forecast'].append({'fid': fid, **ydict})
    
    return result
    
    
def analyzeForecast():
    ...
def featureImportances2(trX):
    
    print("Analyzing Feature Importances...")
    
    #Ts = np.array([r[0] for r in trX[0]])
    trX = [[r[1] for r in x] for x in trX]
    trX = np.array(trX).T
    models = []
    fis = []
    M = trX.T
    for i, trx in enumerate(M):
        #print(i, '/', len(M)) if i % 100 == 0 else None
        trainX = [[v for v in r] for r in trX.T]
        trainX = trainX[:i] + trainX[i+1:]
        trainX = np.array(trainX).T
        
        #print(np.std(trainX))
        regr = RandomForestRegressor(n_estimators=10, max_features='sqrt').fit(trainX, trX[:,i])
        fi = list(regr.feature_importances_)
        
        # regr = BayesianRidge(normalize=True, fit_intercept=True).fit(trainX, trX[:,i])
        # #trainX[:,i] = vec
        # coefs = regr.coef_
        # fi = list(np.abs(coefs)/np.sum(np.abs(coefs)))
        
        
        fi.insert(i,-1)
        models.append(regr)
        fis.append(fi)

    
    print('completed')
        #fis.append(fi)
    #fis = np.array(fis)
    return models, fis

def controlStrategy2(yInd, forecasts, model, control=[0], maximize=True):

    if not control:
        print('No control variables defined, nothing to control...')
        return None, None
    elif control == [-1]:
        control = list(range(len(X)))
    
    print('Figuring out feasible region for output variable...')
    ydict = forecasts['forecast'][yInd]
    tmax = ydict['tNow']
    print('tmax', tmax)
    print('Figuring out feasible regions for input variables...')
    Xdict = forecasts['forecast'][:yInd] + forecasts['forecast'][yInd+1:]

    
    
    Xlow = [np.percentile([v[1] for v in x['pred'] if v[0] <= tmax], 5) for x in Xdict]
    Xhigh = [np.percentile([v[1] for v in x['pred'] if v[0] <= tmax], 95) for x in Xdict]

    tpred = ydict['pred'][-1][0]
    print('Prediction tpred',tpred)

    xbase = np.array([x['pred'][-1][1] for x in Xdict])

    print('Using control variables to define Evolutionary Strategy variance...')
    variance = xbase/10
    #print(variance)
    for i, var in enumerate(variance):
        #print('control', control)
        if i not in control:
            variance[i] = 0
            Xlow[i] = 0
            Xhigh[i] = np.inf
    #print('Xlow', Xlow)
    #print('Xhigh', Xhigh)
    
    if maximize:
        optimize = np.argmax
    else:
        optimize = np.argmin
    
    xopt = xbase
    print('Using Evolutionary Strategy to find optimal control...')
    for iteration in range(5):
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
    tlast = tmax
    xlast = [x['yNow'] for x in Xdict]
    ylast = ydict['yNow']
    print('tlast', tlast)
    #print('Xlast', xlast)
    print('Ylast', ylast)
    print('\n')
    
    ybase = np.clip(model.predict([xbase])[0],0, np.inf)
    print('tbase', tpred)
    #print('xbase', xbase)
    print('ybase', ybase)
    print('\n')
    
    topt = tpred
    print('topt', tpred)
    #print('xopt', xopt)
    print('yopt', yopt)
    
    result = {'last':{'t': tlast, 'X':xlast, 'y':ylast},
              'base':{'t':tpred, 'X':list(xbase), 'y':ybase},
              'opt':{'t':tpred, 'X': list(xopt), 'y':yopt}}
    
    return result

def controlStrategiesRandom(forecasts, models, fids):
    y = random.choice(forecasts['forecast'])
    targetIndex = forecasts['forecast'].index(y)
    maximization = round(random.random())
    mode = 'max' if maximization else 'min'
    
    model = models[targetIndex]
    
    controlVars = random.sample(fids[:targetIndex]+fids[targetIndex+1:], k=random.choice(range(1,4)))
    print('controlVars', controlVars)
    
    controlVarsIndices = []
    newfids = fids[:targetIndex]+fids[targetIndex+1:]
    for cv in controlVars:
        controlVarsIndices.append(newfids.index(cv))
    
    
    cs = controlStrategy2(targetIndex, forecasts, model, control=controlVarsIndices, maximize=maximization)
    
    cs['last']['X'].insert(targetIndex, cs['last']['y'])
    cs['base']['X'].insert(targetIndex, cs['base']['y'])
    cs['opt']['X'].insert(targetIndex, cs['opt']['y'])
    
    print(controlVars)
    result = {'datetime': time.time(), 'mode': mode, 'targetFid': fids[targetIndex], 'controls': controlVars, 'fidsList': fids, **cs}
    return result

def analysis(fis, forecasts, fids, controlStrategies=None):
    
    anomalyIndices = []
    forecasts = forecasts['forecast']
    for ind, forecast in enumerate(forecasts):
        #print('forecast',forecast)
        if random.random() > 0.5: #forecast['isAnomaly']
            anomalyIndices.append(ind)
    
    anomalyFids = [fids[ind] for ind in anomalyIndices]
    
    summary = []
    if anomalyFids:
        for ind, anomalyFid in enumerate(anomalyFids):
            sentence = '指标' + str(anomalyFid) + '出现异常：当前值' + str(np.round(forecasts[ind]['yNow'],2)) + '不在正常范围内' + ' (' + str(np.round(forecasts[ind]['lowNow'],2)) + '~' + str(np.round(forecasts[ind]['highNow'],2))  + ') '
            summary.append(sentence)
        
    if not summary:
        summary.append('无异常')
    
    return summary

def generateJson2(request=None, outputFilename=None, plot=False):
    if request:
        requestJs()
    X, fids, mapping, controlDecision = loadData()
    I = [i for i,x in enumerate(X) if  len(x) > 300] # if 2 < len(x) < 300
    ind = random.choice(I)
    # with open(path / 'data' / 'fake-data' / 'randTimeSeries.json', 'r') as f:
    #     X = json.loads(f.read())
    # ind = random.choice(range(len(X)))
    
    trX = knnRegress(X, n_points=10)
    
    forecasts = forecast2(trX, fids=fids, S=0.5, L=0.5)
    
    forecast = forecasts['forecast'][ind]
    resY = forecast['pred']
    high = forecast['high']
    low = forecast['low']
    
    if plot:
        plt.scatter([r[0] for r in X[ind]], [r[1] for r in X[ind]], s=20, alpha=0.5, c='green')
        #plt.scatter(Ts, trX[:,ind], s=50, c='yellow')
        plt.scatter([r[0] for r in resY], [r[1] for r in resY] , s=10, c='black')
        plt.plot([r[0] for r in resY], [r[1] for r in resY], linewidth=1, c='blue')
        
        plt.axvline(trX[0][-1][0])
        plt.plot([r[0] for r in high], [r[1] for r in high], linewidth=1, c='orange')
        plt.plot([r[0] for r in low], [r[1] for r in low], linewidth=1, c='red')
        
        #plt.plot()
        plt.show()
    
    #coef = C[ind]
    #intercept = I[ind]
    #print('coefs', coef)
    #print('intercept', intercept)
    #print('mag', np.linalg.norm(coef))
    #print('sum', np.sum(coef))
    #plt.scatter(range(len(coef)), coef, s=20, alpha=0.8)
    #plt.axhline()
    #plt.show()
    models, fis = featureImportances2(trX)
    controlStrategy = controlStrategiesRandom(forecasts, models, fids)
    #print(forecasts)
    sentences = analysis(fis, forecasts, fids)
    #print(sentences)
    jsdict = {'datetime': time.time(), 'featureImportances': fis, 'forecasts': forecasts, 'controlStrategies': controlStrategy, 'summary': sentences}
    if not outputFilename:
        outputFilename = 'output.json'
    with open(path / 'data' / 'fake-data' / outputFilename, 'w') as f:
        f.write(json.dumps(jsdict))
    print('JSON done.')

if __name__ == '__main__':

    ######################################
    t1 = time.time()

    generateJson2()
    
    t2 = time.time()
    print('time taken', t2-t1)
    
    
    
    