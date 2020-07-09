# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

with open('./price.txt','r') as f:
    rawdata = f.readlines()
    
    
def exponential_regression (x_data, y_data):
    def func_exp(x, b, c):
        #c = 0
        return (b * x) + c
    print(np.mean(y_data))
    popt, pcov = curve_fit(func_exp, x_data, y_data, p0 = (1, np.mean(y_data)))
    print(popt)
    return func_exp(x_data, *popt)

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

data = [float(v.replace('\n','').split('\t')[1].replace(',','')) for v in rawdata[1:-1]]
data.reverse()
training_set = np.array([[data[i]] for i in range(len(data))])

sc = MinMaxScaler()
training_data = sc.fit_transform(training_set)

seq_length = 200
x, y = sliding_windows(training_data, seq_length)

train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out

num_epochs = 500
learning_rate = 0.01

input_size = 1
hidden_size = 1
num_layers = 1

num_classes = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = criterion(outputs, trainY)
    
    loss.backward()
    
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

lstm.eval()
train_predict = lstm(dataX)

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.show()


#data = [float(v.replace('\n','').split('\t')[1].replace(',','')) for v in rawdata[1:-1]]
#data.reverse()
#data1 = data
#plt.plot(data, alpha=.70)
#
#for i in range(5):
#    trend = exponential_regression(np.array(list(range(len(data1)))),np.array(data1))
#    print(trend)
#    plt.plot(trend)
#    y3 = data1 - trend
#    #plt.plot(y3,c='black')
#    Y = np.fft.fft(y3)
#    #print(Y)
#    np.put(Y, range(100, len(y3)), 0.0)
#    ifft = np.fft.ifft(Y)
#    #print('ifft',ifft)
#    plt.plot(ifft, c='r', alpha=.70)
#    plt.plot(y3-ifft, c = 'y')
#    plt.show()
#    data1 = y3-ifft.real
#
#
#Y = np.fft.fft(data)
#print(Y)
#np.put(Y, range(10, len(data)), 0)
#ifft = np.fft.ifft(Y)
#print('ifft',ifft)
##plt.plot(np.multiply(np.array([1.00008**t for t in range(len(data))]),np.array(ifft)), alpha=.70)
# 