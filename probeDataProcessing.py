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
probeDataPath = path / 'data' / '数据指标整合_仪器与实验室20.08.11-v3.0.xlsx'

def readData(path=probeDataPath):
    df = pd.read_excel(probeDataPath, sheet_name=None)
    
    sheet2 = df['Sheet2']
    st = sheet2.transpose()
    #print(st[0])
    d = {}
    for row in st:
        
        d[row] = dict(st[row])
    
    stage = d[0]['工艺处理阶段']
    tbr = []
    for r in d:
 
        #print('stage',stage)
        if str(d[r]['工艺处理阶段']) == 'nan':
            if str(d[r]['工艺段分类']) != 'nan':
                d[r]['工艺处理阶段'] = stage
            else:
                tbr.append(r)
        else:    
            stage = d[r]['工艺处理阶段']
            
    for i in tbr:
        d.pop(i)
    #print(d)
    features = []
    result = {}
    for i in d:
        #print('i',i)
        row = d[i]
        result[i] = {}
        for j, col in enumerate(row):
            #print('j',j)
            #print('col',col)
            if j <= 3:
                #print('actual value',row[col])
                result[i][col] = row[col]
                
            else:
                
                if j % 2 == 0:
                    #col is key
                    
                    key = col

                    K = key.replace('\n', ' ')
                    features.append(K)
                    result[i][K] = {}
                    result[i][K]['typical'] = str(row[key]).strip()
                    # else:
                    #     continue
                else:
                    
                    
                    #col is value
                    #print(i,j,'triggered')
                    array = [v.strip() for v in row[col].replace(',','').replace('上限：','').replace('下限：','').split('\n')]
                    
                    for ai, a in enumerate(array):
                        if '(' in a:
                            a1 = float(a.split('(')[0])
                            a2 = float(a.split('(')[1].replace(')',''))
                            a = (a1 + a2) / 2
                            array[ai] = a
                           
                    result[i][K]['range'] = array   
                    
                    if '-' in result[i][K]['typical']:
                        #print(i,j)
                        array = [float(a) for a in result[i][K]['typical'].split('-')]
                        typ = np.mean(array)
                        result[i][K]['typical'] = typ
                        result[i][K]['range'] = array
                    
    #print(result)
    formatted = []
    for row in result:
        formatted.append({})
        for key in result[row]:
            #print(key)
            dc = result[row][key]
            #print(dc)
            if isinstance(dc, dict):
                if dc != {'typical': '/', 'range': ['', '']} and dc != {'typical': np.nan, 'range': ['', '']} and dc != {'typical': '', 'range': ['', '']} and dc != {'typical': 'nan', 'range': ['', '']}:
                    try:
                        dc['typical'] = float(dc['typical'])
                    except ValueError:
                        dc['typical'] = -1000
                    
                    try:
                        dc['range'] = [float(a) for a in dc['range']]
                    except ValueError:
                        dc['range'] = [-1000, -1000]
                    formatted[row][key] = dc
                    
            else:
                formatted[row][key] = dc
    
    formatted[14] = {'工艺处理阶段': '前段',
                     '工艺段分类': '前段2#加药间',
                     'processId': 14,
                     '描述': np.nan,
                     '乙酸钠（m3/h）': {'typical': -1000, 'range': [4.0, 0.0]},
                     '液位(m)': {'typical': 4.0, 'range': [4.1, 0.5], 'freq': 5}}
    
    formatted[22] = {'工艺处理阶段': '后段',
                  '工艺段分类': '后段1#加药间',
                  'processId': 22,
                  '描述': np.nan,
                  'PAC（m3/h）': {'typical': 2, 'range':[2.0, 0.0]},
                  '次氯酸钠（m3/h）': {'typical': 0.12, 'range':[0.12, 0.0]},
                  'PAM（m3/h）': {'typical': 1.5, 'range':[1.5, 0.0]}}
    
    formatted[23] = {'工艺处理阶段': '后段',
                  '工艺段分类': '后段2#加药间',
                  'processId': 23,
                  '描述': np.nan,
                  '乙酸钠（m3/h）': {'typical': 2, 'range': [2.0, 0.0]}}
    
    ######################
    ##Extract Frequency Data
    sheetFreq = df['Freq']
    st = sheetFreq.transpose()
    #print(st[0])
    d = {}
    for row in st:
        d[row] = dict(st[row])
    
    features = [feature.strip().replace('\n',' ') for feature in list(d[0].keys())]
    features.remove('检测指标')
    #features = list(set(features))
    #print(features)
    #print(d[0])
    d[0] = {v.replace('\n',' '): d[0][v] for v in d[0]}
    #print(d[0])
    freqs = {feature: d[0][feature] for feature in features}
    
    for feature in freqs:
        if isinstance(freqs[feature],str):
            f = freqs[feature]
            newf = int(np.mean([float(v) for v in f.split(' or ')]) / 10) * 10
            freqs[feature] = newf
    
    #print(freqs)
    #print(formatted)
    featureList = []
    i = 0
    for process in formatted:
        
        for feature in features:
            if feature in process.keys():
                process[feature]['freq'] = freqs[feature]
                process[feature]['featureId'] = i
                featureList.append({'process': process['工艺段分类'], 'feature': feature, 'featureId': i, 'processId': process['processId']})
                i+=1
            
    result = {}
    for process in formatted:
        processName = process['工艺段分类']
        process.pop('工艺段分类')
        result[processName] = process
    
    #sort result by processId
    sortedResult =  dict(sorted(result.items(), key=lambda item: item[1]['processId']))
    fData = result
    #print(sortedResult.keys())
    print('Data extracted and formatted from sheet successfully ...')
    return fData, freqs, featureList
    
def randomGenerateTimeSeries(fData, freqs, days=300):
    #use various mathematical functions on frequency-configured Data to generate full-range Time Series, return Time-Series.
    
    Freqs = sorted([v for v in list(set(freqs.values())) if v > 0])
    #print(fs)
    recordTimes = {}
    for freq in Freqs:
        totalMins = 60*24*days
        recordTime = [a for a in range(totalMins) if a % freq ==0]
        recordTimes[freq] = recordTime
    
    features = list(freqs.keys())
    timeSeries = {}
    for i, process in enumerate(fData):
        print(i,'/', len(fData)-1, 'Generating data for', process)
        timeSeries[process] = {}
        p = fData[process]
        
        for key in p:
            #print('key',key)
            #print(features)
            if key in features:
                #print('p',p)
                feature = key
                timeSeries[process][feature] = []
                typical = p[feature]['typical']
                rnge = p[feature]['range']
                freq = p[feature]['freq']
                
                if max(rnge) < 0:
                    rnge = None
                if typical < 0:
                    typical = None
                
                if not typical and not rnge:
                    typical = int(random.random()*100)
                    rnge = [0, 100]
                
                elif not typical and rnge:
                    typical = np.random.uniform(min(rnge), max(rnge))
                
                elif not rnge and typical:
                    rnge = [typical*1.5, typical*0.5]
                
                #Generate timeseries
                sigma = random.choice([3,4,7,10,20])
                period1 = np.abs(np.random.normal(400000, 200000))
                phase1 = 500000*random.random() 
                
                period2 = np.abs(np.random.normal(100000, 50000))
                phase2 = 50000*random.random() 
                
                period3 = np.abs(np.random.normal(50000, 10000))
                phase3 = 50000*random.random() 
                
                if typical == 0:
                    typical = 0.1
                            
                if random.random() > 0.8:
                    for t in recordTimes[freq]:
                        x = np.abs(np.round(np.clip(np.random.normal(typical, typical*0.5/sigma), min(rnge), max(rnge)),2))
                        d = (t, x)
                        timeSeries[process][feature].append(d)
                else:
                    for t in recordTimes[freq]:
                        x = np.round(np.clip(np.random.normal(typical, typical*0.5/sigma) + np.random.normal(typical*0.3, typical*0.03)*(np.sin(2*np.pi*t/period1 + phase1)) + np.random.normal(typical*0.1, typical*0.01)*(np.sin(2*np.pi*t/period2 + phase2)) + np.random.normal(typical*0.05, typical*0.001)*(np.sin(2*np.pi*t/period3 + phase3)), min(rnge), max(rnge)),2)
                        d = (t, x)
                        timeSeries[process][feature].append(d)
                        
                    
                
    #To do: embed some functional relationships into the data, morph the data   
    print("writing generated data into 'timeSeries.json' ...")
    with open(path / 'data' / 'fake-data' / 'timeSeries.json', 'w') as f:
        f.write(json.dumps(timeSeries))
    print('done.')
    
    return timeSeries
    ...

    
    
def readTimeSeries():
    print('reading time series data...')
    with open(path / 'data' / 'fake-data' / 'timeSeries.json', 'r') as f:
        timeSeries = json.loads(f.read())
    return timeSeries

def embedRel(timeSeries, inFids, outFid, funcs, fData, freqs, featureList):
    #this only embeds ONE particular relationship.
    #inFids = [ fids of input features]
    #outFid = fid of output feature
    #fData contains information about the bounds of features to be clipped when exceeded.
    #freqs contains frequency of data generation
    #timeSeries is where we look for inFids and outFid as sources and destination.
    #featureList is a fid-lookup dictionary.
    #funcs are functions stacked together in a particular order. In particular, it's a stack of tanhs functions. In theory, almost all smooth functions of two inflexion points can be approximated by two tanhs functions.
    print('Synchronizing time domain...')
    outFeature = [feature for feature in featureList if feature['featureId'] == outFid][0]
    inFeatures = [feature for feature in featureList if feature['featureId'] in inFids]
    
    outProcess = outFeature['process']
    outFea = outFeature['feature']
    
    y = timeSeries[outProcess][outFea]
    X = []
    
    for inFeature in inFeatures:
        inProcess = inFeature['process']
        inFea = inFeature['feature']
        x = timeSeries[inProcess][inFea]
        X.append(x)
    
    #get inputFeatures and outputFeature ranges
    # for inFeature in inFeatures:
    #     x = 
    # square = lambda x: x**2
    # linear = lambda x: x
    # gauss = lambda x: np.exp(-x**2)
    # inverse = lambda x: 1/x
    # exp = lambda x: np.exp(x)
    # log = lambda x: np.log(x)
    #tanh = lambda x: np.tanh(x)
    T = [] 
    for x in X:
        T += [d[0] for d in x]
    
    T = sorted(list(set(T))) #common timestamp
    cT = []
    inXs = []
    for t,dy in y:
        #print(t)
        if t in T:
            # and t % 1000 == 0
            inX = []
            for x in X:
                for dt, dx in x:
                    if t == dt:
                        inX.append(dx)
            if len(inX) == len(X):
                inXs.append(inX)
                cT.append(t)
                
    # print(inXs)
    # print('cT', cT)
    print('Normalizing inputs and output...')
    ys = [dy[1] for dy in y]
    miu = np.mean(ys)
    sigma = np.std(ys) + miu/10
    
    mius = np.mean(inXs, axis=0)
    sigmas = np.std(inXs, axis=0) + mius/10
    
    print(mius, sigmas)
    
    print('Randomizing parameters for underlying relationship...')
    params1 = np.random.uniform(-1,1,len(X))
    params2 = np.random.uniform(-1,1,len(X))
    params3 = np.random.uniform(-1,1,2)
    zs = []
    print('Generating underlying relationship...')
    for dx in inXs:
        z = np.tanh(np.dot(params3, (np.tanh(np.dot(params1, (dx-mius)/sigmas)) , np.tanh(np.dot(params2, (dx-mius)/sigma)))))
        zs.append(z)
    zs = np.array(zs)
    #print('z',zs)
    resY = np.abs(zs*sigma + np.random.normal(miu, miu/10))
    
    
    print('Embedding generated relationship back into original time series...')
    currentY = resY[0]
    outY = []
    for t,dy in y:
        #print(t)
        if t in cT:
            ind = cT.index(t)
            outY.append([t,resY[ind]])
            currentY = resY[ind]
        else:
            outY.append([t,np.abs(np.random.normal(currentY, currentY/30))])
            currentY = currentY
    
    resY = [z[1] for z in outY]
    T = [d[0] for d in y]
    
    plt.scatter(T, resY, s=0.3, alpha=0.5)
    #plt.plot(T,resY, linewidth=0.1)
    plt.show()
    #timeSeries
    #print(outY)
    timeSeries[outProcess][outFea] = outY
    ...
    print("[SUCCESS] Data generation done with random relationship embedded.")
    return timeSeries

def train(timeSeries_train, inFids, outFid):
    print("Pulling inputs and output data from time series...")
    outFeature = [feature for feature in featureList if feature['featureId'] == outFid][0]
    inFeatures = [feature for feature in featureList if feature['featureId'] in inFids]
    
    outProcess = outFeature['process']
    outFea = outFeature['feature']
    
    y = timeSeries[outProcess][outFea]
    X = []
    
    for inFeature in inFeatures:
        inProcess = inFeature['process']
        inFea = inFeature['feature']
        x = timeSeries[inProcess][inFea]
        X.append(x)
        
    maxT = max([x[-1][0] for x in X])
    
    print('Synchronizing multiple time series...')
    T = [] 
    for x in X:
        T += [d[0] for d in x]
    T += [d[0] for d in y]
    T = sorted(list(set(T))) #common timestamp in X
    
    #train-test config
    zxs = []
    step = 100
    trainDays = 1500
    testDays = trainDays*1.5
    trainMinutes = step*trainDays
    trainT = [y[0] for y in y if y[0] <= trainMinutes]
    trainY = [y[1] for y in y if y[0] <= trainMinutes]
    n = len(trainY)
    
    
    resY = []
    resX = []
    print('Polynomial fitting output y...')
    py = np.poly1d(np.polyfit(trainT,trainY,50))
    plt.scatter(trainT,trainY, s=0.1, c='green', alpha=0.5)
    #plt.plot([t for t in range(0,maxT,10)], [py(t) for t in range(0,maxT,10)], linewidth=2, color='blue')
    plt.axvline(x=step*trainDays)
    plt.axvline(x=step*testDays)
    plt.show()
    print('Polynomial fitting input Xs...')
    
    pxs = []
    for x in X:
        px = np.poly1d(np.polyfit([x[0] for x in x if x[0] <= trainMinutes],[x[1] for x in x if x[0] <= trainMinutes],50))
        pxs.append(px)
        plt.scatter([x[0] for x in x],[x[1] for x in x], color='green', s=0.5)
        #plt.plot([t for t in range(0,maxT,10)], [px(t) for t in range(0,maxT,10)], linewidth=2, color='blue')
        plt.axvline(x=step*trainDays)
        plt.axvline(x=step*testDays)
        plt.show()
    
    
    print('maxT',maxT)

    for t in range(0,maxT,step):
        zx = [px(t) for px in pxs]
        zxs.append(zx)
    
    zy = [py(t) for t in range(0,maxT, step)]
    
    plt.scatter([y[0] for y in y],[y[1] for y in y], s=0.1,alpha=0.5, color='green')
    #plt.plot([t for t in range(0,maxT, step)], zy, linewidth=2, color='black')
    print('Training model...')
    reg = BayesianRidge().fit(np.array(zxs)[:trainDays], zy[:trainDays])#MLPRegressor(hidden_layer_sizes=(2,))
    
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
    
    
    print('maxT',maxT)

    for t in range(0,maxT,step):
        zx = [px(t) for px in pxs]
        zxs.append(zx)
    
    zy = [py(t) for t in range(0,maxT, step)]
    xs = [[px(t)  for px in pxs] for t in range(0,maxT, step)]
    print('Predicting output...')
    ys = reg.predict(xs)
    plt.plot([t for t in range(0,maxT, step)], ys, color='yellow')
    pY = np.poly1d(np.polyfit([y[0] for y in y],[y[1] for y in y],50))
    plt.plot([t for t in range(0,maxT, step)],[pY(t) for t in range(0,maxT, step)], linewidth=1, c='black')
    #plt.show()
    plt.axvline(x=step*trainDays)
    plt.axvline(x=step*testDays)
    # pY = np.poly1d(np.polyfit(ys[:1000],zy[:1000],3))
    # zY = [pY(y) for y in ys]
    # plt.plot([t for t in range(0,maxT, 100)], zY, linewidth=1, color='yellow')
    
    return py, pxs
    
    
###########################################################


if __name__ == '__main__':
    fData, freqs, featureList = readData()
    #ts = randomGenerateTimeSeries(fData, freqs, days=300)
    ts = readTimeSeries()
    
    # randomFeature = random.choice(featureList)
    # feature = randomFeature['feature']
    # process = randomFeature['process']
    # #print(process, feature)
    
    # d = ts[process][feature]
    # y = [d[1] for d in d][:100]
    # x = [d[0] for d in d][:100]
    # np.min(y)
    # plt.plot(x,y)
    # plt.axhline(y=0)
    # plt.show()
    ...
    
        
    inFids = [6,9,13]
    outFid = 20
    
    outFeature = [feature for feature in featureList if feature['featureId'] == outFid][0]
    inFeatures = [feature for feature in featureList if feature['featureId'] in inFids]
    
    outProcess = outFeature['process']
    outFea = outFeature['feature']
    # ry = ts[outProcess][outFea]
    # T = [ry[0] for ry in ry]
    # Y = [ry[1] for ry in ry]
    
    # plt.scatter(T,Y, s=1)
    # plt.scatter(T,Y, s=0.5, alpha=0.7, c='red')
    # plt.show()
    timeSeries = embedRel(ts, inFids, outFid, funcs=None, fData=fData, freqs=freqs, featureList=featureList)
    # ry = timeSeries[outProcess][outFea]
    # T = [ry[0] for ry in ry]
    # Y = [ry[1] for ry in ry]

    py, pxs = train(timeSeries,inFids, outFid)

    
    