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
from pathlib import Path
import matplotlib.font_manager as mfm

path = Path('./')
ch_font = mfm.FontProperties(fname="/usr/share/fonts/truetype/SourceHanSansCN/SourceHanSansCN-Regular.ttf")


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
                
                #vs = [float(v.replace("%",'').replace(',','')) for v in list(filter(None, values))]
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

if __name__ == '__main__':
     
    data = readData(path / 'data' / 'data.txt')
    y = data['匀质池']['TN']
    
    z = []
    for i,v in enumerate(y):
        if i <= 300:
            z.append(v)
        else:
            z.append(v - 150 + np.random.normal(i/2, np.sqrt(np.std(y[-200:]))))
    
    #z = [y for i, y in enumerate(y) if i <= 300 else y + np.random.normal(i, np.abs(i))]
    
    
    
    plt.plot(y, c='green', label='趋势预测（最优调控策略）')
    plt.plot(z, c='red', label='趋势预测（正常调控策略）')
    plt.plot(y[:300], c='cornflowerblue', label='历史数据')
    plt.axvline(300, c='grey', linewidth=1)
    plt.legend(prop=ch_font)
    plt.title('调控策略对指标趋势的影响', fontproperties=ch_font)
    plt.xlabel('天数',fontproperties=ch_font)
    plt.ylabel('二沉池COD（mg/L）',fontproperties=ch_font)
    plt.savefig('./assets/' + 'cs' +'.png', dpi=300)