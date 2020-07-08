# -*- coding: utf-8 -*-
import numpy as np
import json
import os
#import matplotlib.pyplot as plt
import time
import wastewater as ww
import treatmentEffect as te
from sklearn.linear_model import BayesianRidge
import joblib
#from sklearn.neural_network import MLPRegressor
#from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor, GradientBoostingRegressor
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.linear_model import MultiTaskElasticNetCV
#from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso, LinearRegression, RidgeCV, Lasso, LassoCV, LassoLars, BayesianRidge, Ridge
#from sklearn.preprocessing import StandardScaler
import random
#import gplearn.genetic as gp
from pathlib import Path
#from sklearn.preprocessing import PolynomialFeatures

path = Path('./')

dict2Array = lambda dictionary : [dictionary[feature] for feature in dictionary.keys()]
array2Dict = lambda array, features: dict(zip(list(features),array))

#trX,trY = map(lambda x: np.array(list(map(dict2Array, x))), [X,Y])

def training(X,Y, modelname):
    if not os.path.isdir(path / 'models'/ modelname):
        os.mkdir(path / 'models'/modelname)
    depVarFile = modelname + '.depVar.json'
    with open(path / 'models'/ modelname/ depVarFile,'r') as f:
        depVars = json.loads(f.read())
    trX,trY = map(lambda x: list(map(dict2Array, x)), [X,Y])
    features = list(X[0].keys())
    #scaler = StandardScaler(trX)
    #scaler.fit(trX)
    #trX = scaler.fit_transform(trX)

    trX,trY = np.array(trX),np.array(trY)
    regrs, Ypreds = [], []
    for outputVar in features:
        dependentVars = depVars[outputVar]
        outInd = features.index(outputVar)
        inInds = list(map(lambda var: features.index(var), dependentVars))
        X1 = list(map(list,list(zip(*list(map(lambda ind: trX[:,ind], inInds))))))
        #print(trY[:,outInd])
        #gp.SymbolicRegressor(init_depth=(1,6),verbose=1,parsimony_coefficient=2, generations=20)
        regr = BayesianRidge().fit(X1, trY[:,outInd])
        print(outputVar, regr.coef_, regr.intercept_)
        
        #regr = BayesianRidge().fit(trX, trY)
        regrs.append(regr)
        Ypred = np.clip(regr.predict(X1),0,1e10)
        Ypreds.append(Ypred)
    Ypred = np.array(Ypreds).T
    trY[trY<1e-4] = 0
    Ypred[Ypred<1e-4] = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        e = np.abs(Ypred - trY)/(trY)
    e[np.isnan(e)] = 0
    e[np.isinf(e)] = 0
    errorByRows = np.mean(e,axis=1)*100
    errorByCols = np.mean(e,axis=0)*100
    
    i=0
    for feature in X[0].keys():
        regr = regrs[i]
        filename = feature + '.model'
        file = path / 'models'/ modelname / filename
        joblib.dump(regr, file)
        i+=1
    
    return Ypred, errorByRows, errorByCols

def testing(X,Y, modelname):
    if not os.path.isdir(path / 'models'/ modelname):
        print('model', modelname, 'not yet created!')
        return
    
    depVarFile = modelname + '.depVar.json'
    with open(path / 'models'/ modelname/ depVarFile,'r') as f:
        depVars = json.loads(f.read())
    
    regrs = []
    for feature in X[0].keys():
        filename = feature + '.model'
        file = path / 'models'/ modelname / filename
        regr = joblib.load(file)
        regrs.append(regr)
    
    
    trX,trY = map(lambda x: list(map(dict2Array, x)), [X,Y])
    features = list(X[0].keys())
    #trX = scaler.fit_transform(trX)
   
    trX,trY = np.array(trX),np.array(trY)
    #Ypred = np.clip(regr.predict(trX),0,1e10) 
    Ypreds = []
    
    for i, outputVar in enumerate(features):
        regr = regrs[i]
        print(i,outputVar, regr.coef_, regr.intercept_)
        dependentVars = depVars[outputVar]
        #outInd = features.index(outputVar)
        inInds = list(map(lambda var: features.index(var), dependentVars))
        X1 = list(map(list,list(zip(*list(map(lambda ind: trX[:,ind], inInds))))))
        #gp.SymbolicRegressor(init_depth=(1,6),verbose=1,parsimony_coefficient=1, generations=20)
        Ypred = np.clip(regr.predict(X1),0,1e10)
        Ypreds.append(Ypred)
        
    Ypred = np.array(Ypreds).T
    Ypred = np.round(Ypred,2)
    trY[trY<1e-4] = 0
    Ypred[Ypred<1e-4] = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        e = np.abs(Ypred - trY)/(trY)
    e[np.isnan(e)] = 0
    e[np.isinf(e)] = 0
    errorByRows = np.mean(e,axis=1)*100
    errorByCols = np.mean(e,axis=0)*100
    
    Ypred = list(map(lambda y: array2Dict(y,features), Ypred))
    
    return Ypred, errorByRows, errorByCols

def predict(x,modelname):
    if not os.path.isdir(path / 'models'/ modelname):
        print('model', modelname, 'not yet created!')
        return
    depVarFile = modelname + '.depVar.json'
    with open(path / 'models'/ modelname/ depVarFile,'r') as f:
        depVars = json.loads(f.read())
    
    regrs = []
    for feature in x.keys():
        filename = feature + '.model'
        file = path / 'models'/ modelname / filename
        regr = joblib.load(file)
        regrs.append(regr)
        
    features = list(x.keys())
    trX = [dict2Array(x)]
    trX = np.array(trX)
    Ypreds = []
    
    for i, outputVar in enumerate(features):
        regr = regrs[i]
        print(i,outputVar, regr.coef_, regr.intercept_)
        dependentVars = depVars[outputVar]
        #outInd = features.index(outputVar)
        inInds = list(map(lambda var: features.index(var), dependentVars))
        X1 = list(map(list,list(zip(*list(map(lambda ind: trX[:,ind], inInds))))))
        #gp.SymbolicRegressor(init_depth=(1,6),verbose=1,parsimony_coefficient=1, generations=20)
        Ypred = np.clip(regr.predict(X1),0,1e10)
        Ypreds.append(Ypred)
        
    Ypred = np.array(Ypreds).T
    Ypred[Ypred<1e-4] = 0
    Ypred = array2Dict(Ypred[0], features)
    return Ypred

def computeError(ypred,yactual):
    #ypred, yactual are two arrays
    trY = np.array(yactual)
    Ypred = np.array(ypred)
    trY[trY<1e-4] = 0
    Ypred[Ypred<1e-4] = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        e = np.abs(Ypred - trY)/(trY+0.001)
    e[np.isnan(e)] = 0
    e[np.isinf(e)] = 0
    errorByCols = e*100
    return errorByCols


if __name__=='__main__':
    t1 = time.time()
    modelname = '厌氧'
    print('TRAINING')

    
    model = te.loadModel(modelname)
    opt = te.loadOpt(modelname)
    #opt['可生化性']['max'] = 100
    X,Y = [],[]
    for i in range(10):
        x = ww.wastewater()
        if random.random() > 0.0:
            x.generateFromOpt(opt)
        else:
            x.simulate(random=True)
        X.append(x.water)
        #print('input wastewater',x.water)
        y = te.simulateTreat(x,modelname)['optEff'].water
        #print(y['COD'])
        Y.append(y)
    
    
        
    Ypred, ebr, ebc = training(X,Y, modelname)
    t2 = time.time()
    ########################
    
    print("\nTESTING")
    X,Y = [],[]
    for i in range(10):
        x = ww.wastewater()
        if random.random() > 0.0:
            x.generateFromOpt(opt)
        else:
            x.simulate(random=True)
        X.append(x.water)
        #print('input wastewater',x.water)
        y = te.simulateTreat(x,modelname)['optEff'].water
        #print(y['COD'])
        Y.append(y)
    
    Ypred, ebr, ebc = testing(X,Y, modelname)

    print('error by rows', ebr)
    print('error by cols', ebc)
    print('mean error', np.mean(ebr), '%')
    print('this sample error', ebr[0], '%')
    for feature in Y[0]:
        print(feature, Y[0][feature], Ypred[0][feature])
        
    t3 = time.time()
    print('training time', t2-t1)
    print('testing time', t3-t2)
    print('total time', t3-t1)