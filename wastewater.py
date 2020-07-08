# -*- coding: utf-8 -*-
import json
import numpy as np
from pathlib import Path

#import treatmentEffect as tE
path = Path('./')

class wastewater:
    def __init__(self, wwdict=dict()):
        self.features = json.loads(open(path / 'config'/'features.json','r').read())
        #self.treatments = json.loads(open(path / 'treatments.json','r').read())
        self.wwdict = wwdict
        #self.water = {feature: 0 for feature in self.features}
        #self.water.update(self.wwdict)
        self.water = dict(wwdict)
        
        js = json.loads(open(path / 'config'/'processData_mergedTreatments.json','r').read())
        data = js['data']
        
        xs = [[float(v) for v in d[:17]] for d in data]
        self.xs = xs
        
        #ys = [d[17:] for d in data]
        #self.ys = ys
        
        self.mius, self.sigmas, self.max = np.mean(xs,axis=0), np.std(xs,axis=0), np.max(xs,axis=0)
        self.mius, self.sigmas, self.max = np.append(self.mius,95), np.append(self.sigmas,5), np.append(self.max,100) #可生化性   
    
    def simulate(self, random=True):
        if random:
            x = np.clip(np.random.normal(self.mius,self.sigmas), 0, self.max)
        else:
            #randomization is less
            arr = np.append(np.median(self.xs,axis=0),95)#可生化性中位数95
            arr = np.random.normal(arr, np.abs(1 + arr/5))
            x = np.clip(arr, 0, self.max) 
            
        d = dict(zip(self.features,x))
        self.water.update(d)
        self.water.update(self.wwdict)
        self.conform()
    
    def conform(self, test=False):
        # TN >= N-NO3 + N-NH3
        # Cond > TDS（600℃）> Cl- + SO42-
        normalState = True
        msg = ''
        if not set(self.water.keys()) == set(self.features):
            print("Incomplete Features Error:","wastewater features are not complete, make sure to call wastewater.simulate() before calling wastewater.conform().")
            normalState = False
            msg += '\n错误：输入不完整'
            return False, msg
        
        if not self.water['TN'] >= self.water['N-NO3'] + self.water['N-NH3']:
            print("Condition Error:", "The condition (TN > N-NO3 + N-NH3) is violated.")
            value = self.water['N-NO3'] + self.water['N-NH3']
            value += np.abs(np.random.normal(0, value/2))
            self.water['TN'] = value
            print('TN condition rectified')
            normalState = False
            msg += '\n错误：TN < NO3 + NH3'
            
        if not self.water['TDS（600℃）'] >= self.water['Cl-'] + self.water['SO42-']:
            print("Condition Error:","The condition ('TDS（600℃）' > Cl- + SO42-) is violated.")
            value = self.water['Cl-'] + self.water['SO42-']
            value += np.abs(np.random.normal(0, value/2))
            self.water['TDS（600℃）'] = value
            print('TDS（600℃） condition rectified')
            normalState = False
            msg += '\n错误：TDS600C < Cl- + SO42-'
            
        if not self.water['Cond'] >= self.water['TDS（600℃）']:
            print('Condition Error:',"The condition (Cond > TDS（600℃）) is violated.")
            value = self.water['TDS（600℃）']
            value += np.abs(np.random.normal(0, value/2))
            self.water['Cond'] = value
            print('Cond condition rectified')
            normalState = False
            msg += '\n错误：Cond < TDS600C'
        
        if normalState:
            #print('wastewater is already normal, nothing to do.')
            msg = '输入正常'
        if test:
            return normalState, msg
    
    def signif(self, x, p):
        x = np.asarray(x)
        x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
        mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
        return np.round(x * mags) / mags
    
    def generateFromOpt(self, optDict):
        optState = False
        while not optState:
            self.simulate(random=True)
            water = {feature: self.signif(np.random.uniform(optDict[feature]['min'], optDict[feature]['max']), 2) for feature in optDict}
            self.water.update(water)
            #self.simulate(water)
            optState, msg = self.conform(test=True)
            #print(msg)
            #print(optState)
            
        self.water.update(self.wwdict)
        

if __name__ == '__main__':
    w = wastewater({"COD":500,"TN":230})
    w1 = wastewater({"COD":500,"TN":230})
    w.simulate()
    print(w.water)
    method_list = [func for func in dir(w1) if callable(getattr(w1, func)) and not func.startswith("__")]