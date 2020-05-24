from StockPredictor import StockPredictor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

n_days = 14
depth = 20
estimators = 5000
avg = 0
n_tests = 100

predictor = StockPredictor("SPY")
acc = 0
for i in range(n_tests):
    acc += predictor.backtestNeuralNet(20, 5)

acc /= n_tests

print("Average accuracy: " + str(acc))


#predictor.plotVolume()
#predictor.plotPercentChange()