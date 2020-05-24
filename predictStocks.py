from StockPredictor import StockPredictor
import matplotlib.pyplot as plt
import torch.nn.functional as F

n_days = 14
depth = 20
estimators = 5000

predictor = StockPredictor('SPY')
#predictor.createRandomForest(n_days, depth, estimators)
#predictor.trainAndValidateRandomForest()
#predictor.trainRandomForest()
#print(predictor.predictRandomForest())

predictor.createNeuralNet(n_days)
predictor.trainAndValidateNeuralNet()
avg = 0
n_tests = 1000
for i in range(n_tests):
    predictor = StockPredictor('NVDA')
    predictor.createNeuralNet(n_days)
    avg += predictor.trainAndValidateNeuralNet()
    
avg /= n_tests
print("Average: " + str(avg))

#predictor.trainNeuralNet()
#out = predictor.predictNeuralNet()
#smax = F.softmax(out, dim=0)
#print(smax)

#predictor.plotVolume()
#predictor.plotPercentChange()