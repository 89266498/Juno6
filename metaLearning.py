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
def generateTrainingData():
    nx=3
    length = int(np.random.uniform(30, 800000))
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
            for t in times:
                x = np.abs(np.round(np.clip(np.random.normal(typical, typical*0.5/sigma), min(rnge), max(rnge)),2))
                d = (t, x)
                timeSeries.append(d)
        else:
            for t in times:
                x = np.round(np.clip(np.random.normal(typical, typical*0.5/sigma) + np.random.normal(typical*0.3, typical*0.03)*(np.sin(2*np.pi*t/period1 + phase1)) + np.random.normal(typical*0.1, typical*0.01)*(np.sin(2*np.pi*t/period2 + phase2)) + np.random.normal(typical*0.05, typical*0.001)*(np.sin(2*np.pi*t/period3 + phase3)), min(rnge), max(rnge)),2)
                d = (t, x)
                timeSeries.append(d)
        X.append(timeSeries)
    return X
    ...

if __name__ == '__main__':
    X = generateTrainingData()
