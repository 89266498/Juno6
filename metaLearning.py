# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore') 
import platform
from pathlib import Path
import random
import numpy as np
import json
import os, sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
from sklearn.linear_model import BayesianRidge, LinearRegression, RidgeCV, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from multiprocessing import Pool
import pickle 
import requests
import progress.bar as progBar
#from progress.bar import IncrementalBar
import base64

path = Path('./')
if platform.system() == 'Windows':
    font = '微软雅黑'
else:
    font = 'WenQuanYi Zen Hei' #'Source Han Sans CN'
#plt.rc('font', family=font)
ch_font = mfm.FontProperties(fname="/usr/share/fonts/truetype/SourceHanSansCN/SourceHanSansCN-Regular.ttf")
#plt.rcParams.update({'font.family': font})
#plt.rcParams['font.family'] = ['Source Han Sans CN']
#font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
#prop = mfm.FontProperties(fname=font_path)
# plt.text(0.5, 0.5, s=u'测试', fontproperties=prop)
# plt.show()
if not os.path.isdir(path / 'data'):
    os.mkdir(path / 'data')

if not os.path.isdir(path / 'data' / 'fake-data'):
    os.mkdir(path / 'data' / 'fake-data')

if not os.path.isdir(path / 'models'):
    os.mkdir(path / 'models')
    
if not os.path.isdir(path / 'assets'):
    os.mkdir(path / 'assets')

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
    bar = progBar.Bar('Regressing data...', max=len(Ts))
    for t in Ts:
        bar.next()
        trx = []
        for x in X:
            ts = (1/(np.abs((np.array([r[0] for r in x]) -  t)) + 0.1))**3 # <= (minT + (t-minT)*1)
            ps = ts/np.sum(ts)
            #print('sum', round(np.sum(ps),2))
            xs = np.array([r[1] for r in x]) # <= (minT + (t-minT)*1)
            rx = np.dot(xs,ps)
            trx.append(rx)
        trX.append(trx)
    bar.finish()
    
    trX = np.array(trX)
    Ts = np.array(Ts)
    
    resX = np.array([list(zip(Ts, x)) for x in trX.T])
    
    #print(resX[0])
    return resX

def forecast2(trX, data, fids, S=0.5, L=0.5, A=0.1):
    #print("Forecasting...")
    
    Ts = np.array([r[0] for r in trX[0]])
    trX = [[r[1] for r in x] for x in trX]
    trX = np.array(trX).T
    s = int(len(trX)*S)
    l = int(len(trX)*L)
    a = int(len(trX)*A)
    F = len(trX[0])
    #Y = trX
    #print(len(Ts))
    regrs = []
    resYs = []
    C = []
    I = []
    upps = []
    lwrs = []
    bar = progBar.Bar("Forecasting features...", max=F)
    for ind in range(F):
        bar.next()
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
        fcX = trX[-s-a:-a, ind]
        
        # maxX = np.percentile(trX[:,ind], (1 - anomalyRate/2)*100)
        # minX = np.percentile(trX[:,ind], anomalyRate*100/2)
        
        #print('forecasting')
        for j in range(l+a):
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
    bar.finish()
    YM = np.array(resYs).T
    trX = list(trX[:-a]) + list(YM)
    trX = np.array(trX)

    for j in range(l):  
        Ts = np.append(Ts, Ts[-1] + (Ts[1]-Ts[0]))
    
    Ts1 = Ts[-l-a:]

    resX = [list(zip(list(Ts), list(x))) for x in trX.T]
    upps = [list(zip(list(Ts1), list(x))) for x in upps]
    lwrs = [list(zip(list(Ts1), list(x))) for x in lwrs]
    #print(len(Ts))
    #print(len(trX))  
    
    ydicts = []
    for i, yout in enumerate(resX):

        dt = data[i]
        tPrev = Ts[-l-a] 
        tNow = Ts[-l]
        yNow = yout[-l][1]
        highNow = np.max([upps[i][0][1], upps[i][a][1]])*1
        lowNow = np.min([lwrs[i][0][1], lwrs[i][a][1]])*1
        # highNow = np.max([r[1] for r in dt if r[0] <= tPrev])*1.5
        # lowNow = np.min([r[1] for r in dt if r[0] <= tPrev])*0.7
        
        anomalies = [x for x in dt if (tPrev <= x[0] <= tNow) and not (lowNow <= x[1] <= highNow)]
        total = len([x for x in dt if (tPrev <= x[0] <= tNow)]) + 1
        anomalyRate = round(len(anomalies)/total,4)
        #print('ar', anomalyRate)
        k=100
        yout = [v for j, v in enumerate(yout) if j % int(len(yout)/(k*(a+l)/(1+l))) == 0]
        upp = [v for j, v in enumerate(upps[i]) if j % int(len(upps[i])/k) == 0]
        lwr = [v for j, v in enumerate(lwrs[i]) if j % int(len(lwrs[i])/k) == 0]
        
        ydict = {'pred':yout, 'high': upp, 'low': lwr, 'anomalyRate': anomalyRate, 'anomalies': anomalies, 'highNow': highNow, 'lowNow': lowNow, 'yNow': yNow, 'tPrev': tPrev, 'tNow': tNow, 'pics': [] }
        ydicts.append(ydict)
    
    result = {'datetime':time.time(), 'forecast': [], 'summary': []}
    for i, ydict in enumerate(ydicts):
        fid = fids[i]
        result['forecast'].append({'fid': fid, **ydict})
    
    return result
    

def featureImportances2(trX, fids, threshold=0.01):
    
    #print()
    
    #Ts = np.array([r[0] for r in trX[0]])
    trX = [[r[1] for r in x] for x in trX]
    trX = np.array(trX).T
    models = []
    fis = []
    M = trX.T
    #bar = IncrementalBar('Countdown', max = len(M))
    #print(bar)
    bar = progBar.Bar("Analyzing Feature Importances...", max=len(M))
    for i, trx in enumerate(M):
        #print('\r',i, '/', len(M)) if i % 100 == 0 else None
        bar.next()
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
    bar.finish()    
    
    #print('completed')
        #fis.append(fi)
    #fis = np.array(fis)
    
    #formatting to dict with fids as keys
    sums = [round(sum(fi), 2) for fi in fis]
    #print(sums)
    d = {fid1: {fid2: fis[i][j] for j, fid2 in enumerate(fids) if fis[i][j] >= threshold} for i, fid1 in enumerate(fids)}
    
    return models, d

def controlStrategy2(yInd, forecasts, model, control=[0], maximize=True, iterations=10):
    
    if not control:
        print('No control variables defined, nothing to control...')
        return None
    elif control == [-1]:
        control = list(range(len(forecasts)))
    
    print('Figuring out feasible region for output variable...')
    ydict = forecasts['forecast'][yInd]
    tmax = ydict['tNow']
    print('tmax', tmax)
    print('Figuring out feasible regions for input variables...')
    Xdict = forecasts['forecast'][:yInd] + forecasts['forecast'][yInd+1:]

    
    
    Xlow = [np.percentile([v[1] for v in x['pred'] if v[0] <= tmax], 1) for x in Xdict]
    Xhigh = [np.percentile([v[1] for v in x['pred'] if v[0] <= tmax], 99) for x in Xdict]

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
    bar = progBar.Bar('Evolving solutions...', max=iterations)
    for iteration in range(iterations):
        #print('iteration', iteration)
        bar.next()
        xin = np.array([np.random.normal(xopt, variance) for i in range(30)] + [xopt])
        #print('xinput',xin)
        xin = [np.clip(x, Xlow, Xhigh) for x in xin]

        yout = np.clip(model.predict(xin), 0, np.inf)
        #print('yout',yout)
        
        tolerance = np.std(yout)/np.mean(yout)
        #print('tolerance', tolerance)
        ind = optimize(yout)
        xopt1 = xin[ind]
        dx = np.mean(np.abs(np.array(xopt) - np.array(xopt1)))/np.mean(xopt)
        yopt = yout[ind]
        xopt = xopt1
        #print('dx', dx)
        variance *= dx*10*(iteration+1)
        #print('variance', variance)
        if tolerance < 0.0001:
            print()
            print('Best solution found by convergence...')
            break
    bar.finish()
    #print('control', control)
    tlast = tmax
    xlast = [x['yNow'] for x in Xdict]
    xlast = [v for i, v in enumerate(xlast) if i in control]
    ylast = ydict['yNow']
    #print('tlast', tlast)
    #print('Xlast', xlast)
    #print('Ylast', ylast)
    #print('\n')
    
    ybase = np.clip(model.predict([xbase])[0],0, np.inf)
    xbase = [v for i, v in enumerate(xbase) if i in control]
    #print('tbase', tpred)
    #print('xbase', xbase)
    #print('ybase', ybase)
    #print('\n')
    
    topt = tpred
    xopt = [v for i, v in enumerate(xopt) if i in control]
    #print('topt', tpred)
    #print('xopt', xopt)
    #print('yopt', yopt)
    
    result = {'last':{'t': tlast, 'X':xlast, 'y':ylast},
              'base':{'t':tpred, 'X':list(xbase), 'y':ybase},
              'opt':{'t':tpred, 'X': list(xopt), 'y':yopt}}

    
    #print(result)
    return result

def controlStrategiesRandom(forecasts, models, fids):
    y = random.choice(forecasts['forecast'])
    targetIndex = forecasts['forecast'].index(y)
    maximization = round(random.random())
    mode = 'max' if maximization else 'min'
    
    model = models[targetIndex]
    
    controlVars = random.sample(fids[:targetIndex]+fids[targetIndex+1:], k=random.choice(range(1,4)))
    #print('controlVars', controlVars)
    
    controlVarsIndices = []
    newfids = fids[:targetIndex]+fids[targetIndex+1:]
    for cv in controlVars:
        controlVarsIndices.append(newfids.index(cv))
    print('cv', controlVars)
    print('cvi',controlVarsIndices)
    cs = controlStrategy2(targetIndex, forecasts, model, control=controlVarsIndices, maximize=maximization)
    
    # cs['last']['X'].insert(targetIndex, cs['last']['y'])
    # cs['base']['X'].insert(targetIndex, cs['base']['y'])
    # cs['opt']['X'].insert(targetIndex, cs['opt']['y'])
    
    suggestion = []
    if not cs:
        suggestion = ['目前没定义任何调控变量，所以暂无调控策略。']
    else:
        statement =  '根据当前情况，算法猜测人类根据以往的经验会倾向于进行这样的调控：'
        ss = []
        # print(cs['last']['X'])
        # print(cs['base']['X'])
        # print(len(cs['base']['X']))
        # print('cv', controlVars)
        for i, v in enumerate(cs['base']['X']):
            # print(cs['last']['X'][i])
            # print(cs['base']['X'][i])
            # print(controlVars[i]  +  ' 从 ' + str(round(cs['last']['X'][i],2)) + ' 调节至 ' + str(round(cs['base']['X'][i],2)) )
            ss.append( controlVars[i]  +  ' 从 ' + str(round(cs['last']['X'][i],2)) + ' 调节至 ' + str(round(cs['base']['X'][i],2)) )
            print('ss', ss)
        statement += ('，').join(ss) + '。'
        statement += '预计该调控策略将于' + str(time.ctime(cs['base']['t'])) + '使' + fids[targetIndex] +  '调节至' + str(round(cs['base']['y'],2)) + '。'
        suggestion.append(statement)
        ############
        statement += '算法通过深度模型能给出最优的调控策略是：'
        ss = []
        for i, v in enumerate(cs['opt']['X']):
            ss.append( controlVars[i]  +  '从' + str(round(cs['last']['X'][i],2)) + '调节至' + str(round(cs['opt']['X'][i],2)) )
        statement += ('，').join(ss) + '。'
        statement += '预计该算法给出的调控策略将于' + str(time.ctime(cs['base']['t'])) + '使' + fids[targetIndex] +  '调节至' + str(round(cs['opt']['y'],2)) + '。'
        suggestion.append(statement)
    
    
    
    
    #print('controls',controlVars)
    result = {'datetime': time.time(), 'mode': mode, 'targetFid': fids[targetIndex], 'controls': controlVars, **cs, 'suggestion': suggestion, 'pics': []}
    return result

def analysis(X, fis, forecasts, fids, mapping, controlStrategies=None):
    
    anomalyIndices = []
    forecasts = forecasts['forecast']
    for ind, forecast in enumerate(forecasts):
        #print('forecast',forecast)
        if forecast['anomalyRate'] > 0.1 and len(forecast['anomalies']) > 3:
            #print(forecast['anomalyRate'])
            anomalyIndices.append(ind)
    
    anomalyFids = [fids[ind] for ind in anomalyIndices]
    
    anomalyDicts = []
    if anomalyFids:
        bar = progBar.Bar('Analyzing anomalies...', max=len(anomalyFids))
        for ind in anomalyIndices:
            bar.next()
            anomalyFid = fids[ind]
            #print(len(forecasts[ind]['anomalies']))
            #print(forecasts[ind]['anomalyRate'])
            
            seriousness = round(forecasts[ind]['anomalyRate']*100,2)
            uppNow = np.round(forecasts[ind]['highNow'],2)
            lowNow = np.round(forecasts[ind]['lowNow'],2)
            valNow = np.round(forecasts[ind]['anomalies'][-1][1],2)
            description = '异常情况严重性 ' + str(round(forecasts[ind]['anomalyRate']*100,2)) + '% : ' '指标' + str(anomalyFid) + '最近时间内的值 (' + str(np.round(forecasts[ind]['anomalies'][-1][1],2)) + ') 不在预期安全范围内' + ' (' + str(np.round(forecasts[ind]['lowNow'],2)) + '~' + str(np.round(forecasts[ind]['highNow'],2))  + ') '
            #anomalyDicts.append(description)
            
            fi = fis[anomalyFid]
            sfi = sorted(fi.items(), key=lambda d: d[1], reverse=True)
            
            causes = [fi for fi in sfi if fi[0] in anomalyFids]
            #causes = sfi[:5]
            #print(causes)
            statements = []
            for f in sfi:
                state = '异常' if f in causes else '正常'
                statement = '(' + f[0] + ',' + str(round(f[1], 2)) + ',' + state + ')'
                statements.append(statement)
            
            if statements:
                reasons = ['该异常指标的影响因子（按概率来排序）为：' +  ('，').join(statements)]
                if not causes:
                    reasons.append(['可能原因：设备故障'])
                else:
                    causeFids = [r[0] for r in causes]
                    reasons.append(['可能原因：由' + ('，').join(causeFids) + '出现异常所导致'])
            
            else:
                reasons = ['可能原因：未知']
            pics = []
            if len(causes) >= 1:
                #print('triggered')
                arr = [r[1] for r in sfi[:10]] + [1 - sum([r[1] for r in sfi[:10]])]
                causeFids.append('其它因素')
                labels = [r[0] for r in sfi[:10]] + ['其它因素']
                plt.figure(figsize=(16,9), dpi=300)
                plt.style.use('seaborn')
                #plt.text(fontproperties=prop)
                patches, texts, autotexts = plt.pie(x=arr, labels=labels, autopct='%1.1f%%')
                plt.setp(autotexts, fontproperties=ch_font)
                plt.setp(texts, fontproperties=ch_font)
                leg = plt.legend(prop=ch_font ,facecolor='white', framealpha=1) #bbox_to_anchor=(1, 0, 0.5, 1)
                
                leg.set_title('因素',prop=ch_font)
                plt.title(anomalyFid + '的重要影响因子', fontproperties=ch_font)
                
                #plt.figure(figsize=(16,9))
                #plt.show()
                picname = str(anomalyFid) + '-pie.jpg'
                plt.savefig(path / 'assets' / picname,dpi=100)
                plt.close()
                with open(path / 'assets' / picname, 'rb') as f:
                    base64Data = base64.b64encode(f.read())
                #print(base64Data)
                ab64 = str(base64Data)
                pics.append(ab64)
            ###################################
            forecast = forecasts[ind]
            resY = forecast['pred']
            high = forecast['high']
            low = forecast['low']
            plt.figure(figsize=(16,9))
            plt.style.use('seaborn')
            plt.scatter([r[0] for r in X[ind]], [r[1] for r in X[ind]], s=10, alpha=0.7, c='green', label='历史数据')
            #plt.scatter(Ts, trX[:,ind], s=50, c='yellow')
            #plt.scatter([r[0] for r in resY], [r[1] for r in resY] , s=10, c='black')
            plt.plot([r[0] for r in resY if r[0] <= forecast['tPrev']], [r[1] for r in resY if r[0] <= forecast['tPrev']], linewidth=1, c='blue', label='机器拟合')
            t = [r[0] for r in resY if r[0] <= forecast['tPrev']][-1]
            plt.plot([r[0] for r in resY if r[0] >= t], [r[1] for r in resY if r[0] >= t], linewidth=3, c='cornflowerblue', label='趋势预测')
            plt.axvline(forecast['tPrev'], c='teal')
            plt.axvline(forecast['tNow'], c='teal')
            plt.plot([r[0] for r in high], [r[1] for r in high], linewidth=1, c='orange', label='趋势预测99%置信度上限', ls='dashed')
            plt.plot([r[0] for r in low], [r[1] for r in low], linewidth=1, c='red', label='趋势预测99%置信度下限', ls='dashed')
            anomalies = forecast['anomalies']
            #print(anomalies)
            plt.scatter([r[0] for r in anomalies], [r[1] for r in anomalies], s=20, c='red', label='最近异常点', marker='x')
            plt.title(anomalyFid + '的趋势预测和异常点', fontproperties=ch_font)
            plt.legend(loc="center left", prop=ch_font,facecolor='white', framealpha=1)
            
            #plt.show()
            picname = str(anomalyFid) + '-plot.jpg'
            plt.savefig(path / 'assets' / picname, dpi=100)
            plt.close()
            with open(path / 'assets' / picname, 'rb') as f:
                base64Data = base64.b64encode(f.read())
            #print(base64Data)
            fb64 = str(base64Data)
            pics.append(fb64)
                
            d = {'fid': anomalyFid, 'seriousness': seriousness, 'valNow': valNow, 'lowNow': lowNow, 'uppNow': uppNow, 'description': description, 'causes': sfi, 'reasons': reasons, 'pics': pics}
            anomalyDicts.append(d)
    bar.finish() 
    result = {}
    result['summary'] = []
    if not anomalyDicts:
        result['summary'].append('无异常情况')
        result['anomalies'] = anomalyDicts
    else:
        
        ars = [forecasts[ind]['anomalyRate'] for ind in anomalyIndices]
        seriousness = round(np.median(ars)*100, 2)
        result['summary'].append('发现' + str(len(anomalyDicts)) + '项指标出现异常，总体严重性 ' + str(seriousness) + '%')
        result['anomalies'] = anomalyDicts
    
    return result

def generateJson2(request=None, outputFilename=None, plot=False, fake=False):
    if request:
        requestJs()
        
    if fake:
        data = readData()
        X, fids, mapping, controlDecision = loadData()
        
        fids = random.sample(fids, len(data))
        #print(fids)
        X = data
        
    else:
        X, fids, mapping, controlDecision = loadData()
    
        X = X
        fids = fids
        

        
    I = [i for i,x in enumerate(X) if  len(x)] # if 2 < len(x) < 300
    ind = random.choice(I)
    # with open(path / 'data' / 'fake-data' / 'randTimeSeries.json', 'r') as f:
    #     X = json.loads(f.read())
    # ind = random.choice(range(len(X)))
    
    trX = knnRegress(X, n_points=200)
    
    forecasts = forecast2(trX, data=X, fids=fids, S=0.5, L=0.5)
    
    bar = progBar.Bar('Plotting forecasts...', max=len(I))
    for ind in I:
        bar.next()
        forecast = forecasts['forecast'][ind]
        resY = forecast['pred']
        high = forecast['high']
        low = forecast['low']
        if plot:
            plt.scatter([r[0] for r in X[ind]], [r[1] for r in X[ind]], s=5, alpha=0.5, c='green')
            #plt.scatter(Ts, trX[:,ind], s=50, c='yellow')
            #plt.scatter([r[0] for r in resY], [r[1] for r in resY] , s=10, c='black')
            plt.plot([r[0] for r in resY], [r[1] for r in resY], linewidth=1, c='blue')
            plt.axvline(forecast['tPrev'])
            plt.axvline(forecast['tNow'])
            plt.plot([r[0] for r in high], [r[1] for r in high], linewidth=1, c='orange')
            plt.plot([r[0] for r in low], [r[1] for r in low], linewidth=1, c='red')
            anomalies = forecast['anomalies']
            #print(anomalies)
            plt.scatter([r[0] for r in anomalies], [r[1] for r in anomalies], s=50, c='red')
            #plt.axhline(forecast['highNow'])
            #plt.axhline(forecast['lowNow'])
            #plt.plot()
            #plt.show()
            pic = fids[ind] + '.png'
            plt.savefig(path / 'assets' / pic)
            plt.close()
    bar.finish()
    #coef = C[ind]
    #intercept = I[ind]
    #print('coefs', coef)
    #print('intercept', intercept)
    #print('mag', np.linalg.norm(coef))
    #print('sum', np.sum(coef))
    #plt.scatter(range(len(coef)), coef, s=20, alpha=0.8)
    #plt.axhline()
    #plt.show()
    models, fis = featureImportances2(trX, fids)
    controlStrategy = controlStrategiesRandom(forecasts, models, fids)
    #print(forecasts)
    analyses = analysis(X, fis, forecasts, fids, mapping)
    #print(sentences)
    print(analyses['summary'])
    jsdict = {'datetime': time.time(), 'featureImportances': fis, 'forecasts': forecasts, 'controlStrategies': controlStrategy, 'analysis': analyses}
    if not outputFilename:
        outputFilename = 'output.json'
    with open(path / 'data' / 'fake-data' / outputFilename, 'w') as f:
        f.write(json.dumps(jsdict))
    print('JSON done.')
    return jsdict

if __name__ == '__main__':

    ######################################
    t1 = time.time()

    result = generateJson2(plot=False, fake=True)
    
    t2 = time.time()
    print('time taken', t2-t1)
    
    
    
    