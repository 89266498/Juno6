# -*- coding: utf-8 -*-
from pathlib import Path
import random
import numpy as np
import json
import os
import time
import pandas as pd

path = Path('./')
probeDataPath = path / 'data' / '数据指标整合_仪器与实验室20.08.11初稿.xlsx'

def readData(path=probeDataPath):
    df = pd.read_excel(probeDataPath, sheet_name=None)
    
    sheet2 = df['Sheet2']
    st = sheet2.transpose()
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
    
    result[22] = {'工艺处理阶段': '后段',
                  '工艺段分类': '1#加药间',
                  'processId': 23,
                  '描述': np.nan,
                  'PAC（m3/h）': {'typical': 2, 'range':[2.0, 0.0]},
                  '次氯酸钠（m3/h）': {'typical': 0.12, 'range':[0.12, 0.0]},
                  'PAM（m3/h）': {'typical': 1.5, 'range':[1.5, 0.0]}}
    
    result[23] = {'工艺处理阶段': '后段',
                  '工艺段分类': '2#加药间',
                  'processId': 24,
                  '描述': np.nan,
                  '乙酸钠（m3/h）': {'typical': 2, 'range': [2.0, 0.0]}}

    print('Data extracted and formatted from sheet successfully ...')
    return formatted

def frequentize(formattedData):
    #TO DO: to add frequency configuration of data generation into each key (feature) for each process in formattedData
    ...
    
def randomGenerateTimeSeries(fData):
    #use various mathematical functions on frequency-configured Data to generate full-range Time Series, return Time-Series.
    ...

if __name__ == '__main__':
    result = readData()
