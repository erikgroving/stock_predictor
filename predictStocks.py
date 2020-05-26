from StockPredictor import StockPredictor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

n_days = 14

# random forest parameters
depth = 20
estimators = 5000

#LSTM params
input_size = 2 # volume + % change
hidden_size = 512
sequence_length = 10
num_layers = 1
daysForTraining = 100

sum = 0
n_tests = 10

predictor = StockPredictor("SPY")
for i in range(n_tests):
    sum += predictor.backtestLstm(input_size, hidden_size, num_layers, daysForTraining, sequence_length) 

acc = sum / n_tests
print('Accuracy: ', str(acc))

#for i in range(n_tests):
#    acc += predictor.backtestNeuralNet(20, 5)

#predictor.plotVolume()
#predictor.plotPercentChange()