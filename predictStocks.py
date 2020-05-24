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
predictor.backtestNeuralNet(50, 14)
#predictor.plotVolume()
#predictor.plotPercentChange()