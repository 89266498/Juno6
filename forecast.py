# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import progress.bar as progBar
from sklearn.linear_model import BayesianRidge , LassoCV, RidgeCV
from pathlib import Path

path = Path('./')

def knnRegress(X, n_points=300):
    #for downsampling, denoising and synchronizing signals
    T = []
    for x in X:
        T += [r[0] for r in x]
      
    T = list(set(T))
    minT = int(min(T))
    maxT = int(max(T))
    
    Ts = range(minT, maxT+0*int((maxT-minT)/n_points), int((maxT-minT)/n_points))
    #Ts = sorted(random.sample(range(minT, maxT), n_points))
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

def forecast2(trX, data, S=0.5, L=0.5, A=0.1, absolute=True, plot=True, style='dark_background', train=0.5):
    #print("Forecasting...")
    X = data
    Ts = np.array([r[0] for r in trX[0]])
    trX = [[r[1] for r in x] for x in trX]
    trX = np.array(trX).T
    trX = trX[:int(len(trX)*train)]
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
        regr = RidgeCV(normalize=True, fit_intercept=True).fit(trainX, trainY)  #RidgeCV(normalize=True, alphas=[1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]), LassoCV(normalize=True, fit_intercept=False, alphas=10**np.array(list(range(2,8))))
        
        
        coefs = regr.coef_
        intercept = regr.intercept_
        #plt.plot(coefs)
        #plt.show()
        
        iterations = 20
        offsprings = 10
        
        
        for iteration in range(iterations):
            print('iteration', iteration+1, 'out of', iterations)
            cs = [np.random.normal(coefs, np.abs(coefs/10)) for i in range(offsprings)] + [coefs]
            intercepts = [np.random.normal(intercept, 1*np.abs(intercept/10)) for i in range(offsprings)] + [intercept]
            #j = random.choice(range(len(trainX)))
            fcYs = [[np.dot(cs[k], x) + intercepts[k] for x in trainX] for k,c in enumerate(cs)]
            #print('predicted', fcYs)
            errs = [np.mean(np.abs(trainY - np.array(fcY))) + 0*np.abs(np.linalg.norm(cs[k])-0) + 0*intercepts[k] for k, fcY in enumerate(fcYs)]
            print('errs',errs)
            
            j = np.argmin(errs)
            
            coefs = cs[j]
            intercept = intercepts[j]
            #plt.plot(coefs)
            #plt.show()
        
        #g = lambda x: p(x)/np.sqrt(np.sum(coefs**2)) - intercept/np.sqrt(np.sum(coefs**2)) + intercept
        #print('intercept', intercept)
        
        #coefs = coefs / np.linalg.norm(coefs)
        #coefs = np.clip(coefs, -np.inf,np.percentile(coefs, 98)) 
        #plt.plot(coefs)
        #plt.show()
        print('mag',np.linalg.norm(coefs))
        print('coefs', np.round(coefs,2))
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
            #fcY, std = regr.predict([fcX], return_std=True)#, return_std=True
            #fcY = fcY/np.sqrt(np.sum(coefs**2)) - intercept/np.sqrt(np.sum(coefs**2)) + intercept
            #fcY = regr.predict([fcX])
            #std = fcY*0.1
            fcY = [np.dot(coefs, fcX) + intercept]
            #print(fcY)
            #print(std)
            #Y = np.append(Y, fcY[0])
            fcX =  np.append(fcX, fcY[0])
            if absolute:
                upp.append(list(np.clip(fcY[0] + 3*std, 0, np.inf))[0])
                lwr.append(list(np.clip(fcY[0] - 3*std, 0, np.inf))[0])
                resY.append(np.clip(fcY[0], 0 , np.inf))
            else:
                upp.append(list(fcY[0] + 3*std)[0])
                lwr.append(list(fcY[0] - 3*std)[0])
                resY.append(fcY[0])
            fcX = fcX[-s:]
            #trX = trX.T
            

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
    bar = progBar.Bar('Generating results...', max=len(resX))
    
    for i, yout in enumerate(resX):
        bar.next()
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
        #k=100
        # yout = [v for j, v in enumerate(yout) if j % int(len(yout)/(k*(a+l)/(1+l))) == 0]
        # upp = [v for j, v in enumerate(upps[i]) if j % int(len(upps[i])/k) == 0]
        # lwr = [v for j, v in enumerate(lwrs[i]) if j % int(len(lwrs[i])/k) == 0]
        
        upp = upps[i]
        lwr = lwrs[i]
        if plot:
            plt.figure(figsize=(16,9))
            plt.style.use(style)
            plt.scatter([r[0] for r in dt], [r[1] for r in dt], s=10, alpha=0.7, c='green', label='Historical Data')
            #plt.scatter(Ts, trX[:,ind], s=50, c='yellow')
            #plt.scatter([r[0] for r in resY], [r[1] for r in resY] , s=10, c='black')
            plt.plot([r[0] for r in resX[i] if r[0] <= tPrev], [r[1] for r in resX[i] if r[0] <= tPrev], linewidth=1, c='blue', label='Regression Line')
            t = [r[0] for r in resX[i] if r[0] <= tPrev][-1]
            plt.plot([r[0] for r in resX[i] if r[0] >= t], [r[1] for r in resX[i] if r[0] >= t], linewidth=3, c='cornflowerblue', label='Forecast')
            plt.axvline(tPrev, c='teal')
            plt.axvline(tNow, c='teal')
            plt.axvline(Ts[int(len(Ts)*train)], c='teal')
            #plt.plot([r[0] for r in upps], [r[1] for r in upps], linewidth=1, c='orange', label='99% Confidence Interval Upper Bound', ls='dashed')
            #plt.plot([r[0] for r in lwrs], [r[1] for r in lwrs], linewidth=1, c='red', label='99% Confidence Interval Lower Bound', ls='dashed')
            
            #print(anomalies)
            plt.scatter([r[0] for r in anomalies], [r[1] for r in anomalies], s=20, c='red', label='Recent Anomalies', marker='x')
            plt.title('Forecast and Anomalies')
            plt.legend(loc="center left") #,facecolor='white', framealpha=1
            
            #plt.show()
            picname = str(i) + '-forecast.jpg'
            plt.savefig(path / picname, dpi=300)
        
        
        ydict = {'pred':yout, 'high': upp, 'low': lwr, 'anomalyRate': anomalyRate, 'anomalies': anomalies, 'highNow': highNow, 'lowNow': lowNow, 'yNow': yNow, 'tPrev': tPrev, 'tNow': tNow, 'pics': [] }
        ydicts.append(ydict)
    bar.finish()
    
    return ydicts

def predict(X, n_points=200, **kwargs):
    trX = knnRegress(X, n_points=n_points)
    result = forecast2(trX, data=X, **kwargs)
    return result
    

#######################
if __name__ == '__main__':
    with open(path / 'price.txt', 'r') as f:
        prices = [float(d.strip().split('\t')[-1].replace(',','')) for d in f.readlines()[1:] if d.strip()]
    prices.reverse()
    
    regr = BayesianRidge(normalize=True, fit_intercept=True).fit(np.array(range(len(prices))).reshape(-1,1), np.array(prices).reshape(-1,1))
    p = lambda x: regr.predict(x, return_std=True)
    coefs = regr.coef_
    intercept = regr.intercept_
    
    g = lambda x: p(x)/np.sqrt(np.sum(coefs**2)) - intercept/np.sqrt(np.sum(coefs**2)) + intercept
    
    pred, std = p(np.array(range(len(prices))).reshape(-1,1))
    
    #plt.plot(prices, linewidth=1)
    #plt.plot(pred)
    #plt.plot(pred+3*std, c='orange', linewidth=1)
    #plt.plot(pred-3*std, c='red', linewidth=1)
    #plt.show()
    
    upp = pred + 3*std
    lwr = pred - 3*std
    #print(pred)
    y = np.multiply(3*std, np.tanh(np.divide((prices - pred), 3*std))) + pred
    #y = prices
    #y = np.clip(pred, upp, lwr)
    #plt.plot(y, c='cornflowerblue', linewidth=1)
    #plt.show()
    y=prices
    data = [list(zip(range(len(y)), y))]
    result = predict(data, S=0.1, A=0.9, L=1, train=9.99/10)
        
        
    
    
    
    
   