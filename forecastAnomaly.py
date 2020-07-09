# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import random
import scipy.optimize
from scipy.optimize import curve_fit
from scipy import signal
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
import math
from PIL import Image
import os

def readData(path):
    with open(path, 'r') as f:
        rawdata = f.readlines()
    #print(rawdata)
    if '\n' in rawdata:
        rawdata.remove('\n') #remove last line
    rawdata = [rd.replace('\n','') for rd in rawdata]
    
    data = {}
    #print(rawdata)
    header = list(filter(None, rawdata[0].split('\t')))
    for h in header:
        data[h] = {}
        
    hd = rawdata[0].split('\t')
    
    counts = []
    
    c = 0
    for e in hd:
        if e == '':
            c += 1
        else:
            counts.append(c)
            c = 0 
    counts.append(c)
    counts = counts[1:]
    
    indices = []
    count = 0
    for c in counts:
        count = count + c + 1
        indices.append(count)
    
    indices = [0] + indices
    
    i = 0
    ind = 0
    for d in data:
        features = rawdata[1].split('\t')[indices[i]:indices[i+1]]
        for feature in features:  
            values = [row.split('\t')[ind].strip() for row in rawdata[4:]]
            try:
                vector = [float(v.replace("%",'').replace(',','')) for v in values]
            except ValueError:
                vector = []
                #print(values)
                vs = [float(v.replace("%",'').replace(',','')) for v in list(filter(None, values))]
                #typV = np.mean(vs)
                #print(vs)
                for v in values:
                    if v == '':
                        
                        vector.append(-1)
                    else:
                        vector.append(float(v.replace('%','').replace(',','')))
            data[d][feature] = vector
            ind += 1
        i += 1
    print('data processed')
    return data

def xyFromData(data):
    inflow = ['匀质池', '50%液体葡萄糖', '缺氧池A', '活性污泥池A（ASR）', 'CBR(A)', '缺氧池B', '活性污泥池B（ASR）', 'CBR(B)']
    outflow = '二沉池'
    
    resultIn = {}
    resultOut = {}
    
    for d in data:
        if d in inflow:
            for dd in data[d]:  
                title = d + ':' + dd
                v = data[d][dd]
                if -1 not in v or -1 in v:
                    #print(v[0])
                    resultIn[title] = data[d][dd]
    for dd in data[outflow]:
        title = outflow + ':' + dd
        resultOut[title] = data[outflow][dd]
    
    return resultIn, resultOut

############ Regression
    
def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    #print(1/ff)
    #print(Fyy)
    #print(len(Fyy))
    FF = 1/ff
    print('top 30 periods',sorted(list(FF),reverse=True)[1:30])
    plt.plot( 1/ff,Fyy, c='g', alpha=0.5)
    plt.show()
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset]) #
    k,n = 1,4 #k=dummy var, n=wave number
    def sinfunc(t, c):  return np.multiply(c[:n*k], np.sin(c[n*k:n*(k+1)]*t + c[n*(k+1):n*(k+2)])) + c[n*(k+2):n*(k+3)] #A,w,p,c are numpy arrays
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=np.random.normal(0,1,n*4))
    c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: np.multiply(c[:4*k], np.sin(c[n*k:n*(k+1)]*t + c[n*(k+1):n*(k+2)])) + c[n*(k+2):n*(k+3)]
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def linear_regression (x_data, y_data):
    def func_exp(x, b, c):
        #c = 0
        return (b * x) + c
    #print(np.mean(y_data))
    popt, pcov = curve_fit(func_exp, x_data, y_data, p0 = (y_data[-1] - y_data[0], np.mean(y_data)))
    #print(popt)
    return func_exp(x_data, *popt)

#######################################################
#### Time Series Analysis Tools
    
def autocorrelation(y, plot=True, prune=True):
    corrs = []
    r2s = []
    y1 = y
    for i in range(0,int(0.9*len(y))):
        if i == 0:
            c = np.corrcoef(y1,y)[0][1]
            r2 = r2_score(y1,y)
        else:
            c = np.corrcoef(y[:-i],y1[i:])[0][1]
            r2 = r2_score(y[:-i],y1[i:])
        #print(i, c)
        corrs.append(c)
        if np.isnan(r2):
            r2 = 0
        r2s.append(r2)
    r2s = np.clip(r2s,-5,1)
    #print(r2s)
    if plot:
        plt.plot(corrs, c='g', alpha=0.5)
        plt.plot(r2s)
        plt.title('Autocorrelation and R2 scores')
        plt.show()
    lags = []
    for j in range(len(corrs)):
        if prune:
            if r2s[j] >= min(0.3,np.percentile(r2s,75)) and j > 0 and j != len(corrs)-1:
                lags.append((j,corrs[j],r2s[j]))
        else:
            lags.append((j,corrs[j],r2s[j]))
    return lags

def moving_average(a, n=3) :
    d = math.ceil(n/2)
    res = []
    for i in range(len(a)):
        if i < d:
            ind = 0
        else:
            ind = i
        res.append(np.mean(a[ind:i+d]))
      
    return np.array(res)

def fftApprox(y, thresh=5, plot=True):
    Y = np.fft.fft(y)
    f = np.fft.fftfreq(len(y))
    psd = np.abs(Y)**2/np.sum(np.abs(Y)**2)
    threshold = np.percentile(psd, 100-thresh)
    filtered = np.array([a if a > threshold else 0 for a in psd])
    #print(filtered)
    f[f==0] =  1
    if plot:
        plt.scatter(1/(f),psd, c='black',alpha=0.5)
        plt.axhline(threshold)
        plt.title('PSD-Period')
        plt.show()
    indices = [1 if a > 0 else 0 for a in filtered]
    periods = (1/f)[filtered>0]
    periods = np.round([p for p in periods if p > 0],1)
    #print('filtered periods', periods)
    #np.put(Y, range(cutoff, len(y)), 0.0)
    Y = np.multiply(Y,indices)
    ifft = np.fft.ifft(Y)
    return ifft.real, periods

def featuresAnalysis(data, tank, outputFeature, d1=0, d2=-1, top=5):
    rx,ry = xyFromData(data)
    features = list(rx.keys())
    query = tank+':'+outputFeature
    if d2 == -1:
        X = np.array([rx[d] for d in rx]).T[d1:]
        y = np.array(ry[query])[d1:]
    else:
        X = np.array([rx[d] for d in rx]).T[d1:d2]
        y = np.array(ry[query])[d1:d2]
    
    regressor = DecisionTreeRegressor(random_state=int(10e6*random.random()), max_features='auto')
    scores = cross_val_score(regressor, X, y, cv=10)
    #print('best features', maxf)
    print('GINI FEATURE IMPORTANCES')
    print('final scores',list(scores))
    print('final mean score', np.mean(scores))
    
    
    
    featureImportances = regressor.feature_importances_
    #print(regressor.max_features_)
    sortedFeatures = sorted(featureImportances, reverse=True)
    indices = [list(featureImportances).index(s) for s in sortedFeatures][:5]
    result = []
    #print(indices)
    i = 0
    for ind in indices:
        print(features[ind], sortedFeatures[i])
        i += 1
        result.append((features[ind], sortedFeatures[i]))
    return result


def trendLine(y,n):
    #trend = linear_regression(np.array(list(range(len(y)))),np.array(y))
    trend = moving_average(y, n=n)
    return trend

def analyzeTimeSeries(y, window, plot=True, thresh=3):
    
    n=min(int(len(y)/2),2*window)
    
    lags = autocorrelation(y, plot=plot)
    trend = trendLine(y, n)
    baseline = y - trend
    seasonalLine, periods = fftApprox(baseline, thresh=thresh, plot=plot)
    noise = baseline - seasonalLine
    dev = np.std(noise)
    if plot:
        plt.plot(trend, c='red', linewidth=3)
        plt.plot(y, c='blue', alpha=0.5)
        plt.plot(trend + seasonalLine, c='black')
        plt.plot(trend + seasonalLine + 4*dev, c='teal')
        plt.plot(trend + seasonalLine - 4*dev, c='teal')
        plt.title('Time Series, Trend, Seasonal Pattern')
        plt.show()
        
        plt.plot(noise, c='green')
        plt.axhline(0,linewidth=0.5)
        plt.title('Noise')
        plt.show()
        
        plt.hist(noise)
        plt.title("Noise Profile")
        plt.show()
        
    return lags, trend, trend + seasonalLine, dev, noise, periods

def forecastPredict(y, forecastDays, trainingDays=-1, plot=False, saveFig=False, warning=False, warningHistory=[]):
    if trainingDays < 0:
        trainingDays= len(y)
    lags, trend, baseline, dev, noise, periods = analyzeTimeSeries(y[:trainingDays], window=forecastDays, thresh=1, plot=False)
    forecast = baseline[-1] - baseline[0] + baseline[:forecastDays]
    days = trainingDays
    #print('dev',dev)
    #print('baseline',baseline)

    upperForecast = list(baseline + 4*dev) + list((baseline[:forecastDays] + baseline[-1] - baseline[0])  + 4*dev)
    lowerForecast = list(np.clip(baseline - 6*dev,0,10e10)) + list(np.clip( (baseline[:forecastDays] + baseline[-1] - baseline[0])  - 6*dev, 0, 10e10))
    
    actual = y
    if warning:
        if warningHistory == []:
            warningDay, warningValue = [],[]
        else:
            warningDay, warningValue = warningHistory[0], warningHistory[1]
        for j in range(days,days+forecastDays):
            #print(len(y),len(upperForecast),len(lowerForecast))
            try:
                if y[j] > upperForecast[j] or y[j] < lowerForecast[j]:
                    warningDay.append(j)
                    warningValue.append(y[j])
            except IndexError:
                pass
        warningHistory = [warningDay,warningValue]
        #print(warningHistory)
    if plot:
        if saveFig:
            plt.ioff()
        plt.plot(upperForecast, c='red', linewidth=0.5, label='upper threshold')
        plt.plot(lowerForecast, c='green', linewidth=0.5, label='lower threshold')
        plt.title('forecast & anomaly detection')
        plt.plot(actual, c='blue', linewidth=1, label='actual')
        plt.plot(list(baseline) + list(forecast), c='black', linewidth=1, label='predicted')
        if warning:
            plt.scatter(warningDay, warningValue, marker='o', s=30, c='r', label='anomaly detected')
        plt.legend()
        plt.axvline(days)
        #plt.show()
        if saveFig:
            plt.savefig('./img/' + str(days) +'.png')
            #plt.clf()
        else:
            plt.show()
        plt.clf()
    
    forecast = list(baseline) + list(forecast)      
    prediction = np.array(forecast[days:days+forecastDays])
    actual = np.array(y[days:days+forecastDays])
    try:
        error = np.mean(np.abs((prediction-actual)))
        print('test Mean Absolute Error', error)
    except ValueError:
        error = -1
    
    return forecast, lowerForecast, upperForecast, warningHistory, error

def trainingForecast(y, trainingDays, forecastDays, saveGIF=True):
    file_names = []
    warningHistory = []
    errors = []
    for days in range(trainingDays,len(y),10):
        print('day', days)
        #days = 103
        forecastDays = 30
        lags, trend, baseline, dev, noise, periods = analyzeTimeSeries(y[:days], window=forecastDays, thresh=0.1, plot=False)

        forecast, lowerForecast, upperForecast, warningHistory, error = forecastPredict(y[:days+forecastDays], trainingDays=days, forecastDays=forecastDays, plot=True, warning=True, saveFig=True, warningHistory=warningHistory)
        errors.append(error)
    if saveGIF:  
        print('generating GIF')
        file_names = [fn for fn in os.listdir('./img')]
        frames = []
        I = sorted([int(i.replace('.png','')) for i in file_names])
        filenames = [str(i) +'.png' for i in I]
        for i in filenames:
            
            #print(i)
            new_frame = Image.open('./img/' + i)
            frames.append(new_frame)
         
        # Save into a GIF file that loops forever
        frames[0].save('./training vs forecast.gif', format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=1000, loop=0)
        
        plt.plot(errors)
        plt.title("test error")
        plt.show()
        
    return warningHistory, errors

###################################
if __name__ == '__main__':
    
    data = readData('./data.txt')   
    #featureImportances = featuresAnalysis(data, tank='二沉池', outputFeature='COD')
    y = data['匀质池']['COD']
    
#    if not os.path.isdir('./img'):
#        os.mkdir('./img')
#    else:
#        dirs = os.listdir('./img')
#        [os.remove('./img/' + d) for d in dirs]
#    
#    warningHistory, errors = trainingForecast(y, trainingDays=30, forecastDays=30, saveGIF=True)
#    
#    forecastPredict(y, forecastDays=50, warning=True, warningHistory=warningHistory, plot=True)
#    
    #####################
    
    x1 = data['匀质池']['pH']
    x2 = data['匀质池']['SS']
    x3 = data['匀质池']['Cond']
    
    x4 = data['二沉池']['pH']
    x5 = data['二沉池']['Cond']
    x6 = data['二沉池']['TOC']
    z = data['匀质池']['TOC在线']
    
    z1 = x1+x4 #pH
    
    z2 = x3+x5 #Cond
    z3 = z + x6 #TOC
    
    plt.scatter(z1,z3, alpha=0.5,s=1)
    plt.show()
    
    plt.scatter(z2,z3, alpha=0.5,s=1)
    plt.show()
    
    plt.scatter(z1,z2, alpha=0.5,s=1)
    plt.show()
    
    print(len(z3))
    fig = plt.figure()
    
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(z1[:len(x1)],z2[:len(x1)],z3[:len(x1)], marker='.',alpha=0.5, s=1)
    ax.scatter(z1[len(x1):],z2[len(x1):],z3[len(x1):], marker='.',alpha=0.5, s=1,c='green')
    ax.set_xlabel('pH')
    ax.set_ylabel('Cond')
    ax.set_zlabel('TOC')
    plt.show()
    
    #plt.show()
    
    
    
    
    
    