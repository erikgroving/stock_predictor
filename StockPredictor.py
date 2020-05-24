import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path
from StockDataApi import StockDataApi
from StockDataFetcher import StockDataFetcher
from StockRandomForest import StockRandomForest
from StockNeuralNetwork import StockNeuralNet

from sklearn.preprocessing import normalize


class StockPredictor:
    ticker = ''
    volume = []
    dayChanges = []
    dates = []
    rf = StockRandomForest(14, 4, 1000)

    def __init__(self, ticker):
        self.ticker = ticker

    def createRandomForest(self, days, depth, estimators):
        self.rf = StockRandomForest(days, depth, estimators)
        self.readDataFromJson()
        self.rf.createDataset(self.volume[1:], self.dayChanges[1:])
    
    def trainRandomForest(self):
        print('Training random forest...')
        self.rf.train()

    def trainAndValidateRandomForest(self):
        print('Training and validating random forest...')
        return self.rf.trainAndValidate()


    def predictRandomForest(self):
        point = self.rf.createDataPoint(len(self.volume), self.volume, self.dayChanges)
        point = np.asarray(point).reshape(1, -1)
        output = self.rf.predict(point)[0]
        if output == 1:
            return 'Random forest predicts that ' + self.ticker + ' will be green.'
        else:
            return 'Random forest predicts that ' + self.ticker + ' will be red.'
        return self.rf.predict(point)


    def createLstm(self):
        return

    def createNeuralNet(self, days):
        self.readDataFromJson()
        self.net = StockNeuralNet(days, 30)
        self.normVol = torch.Tensor(np.asarray(normalize([self.volume[1:]])[0]))
        self.normDayChanges = torch.Tensor(np.asarray(normalize([self.dayChanges[1:]])[0]))

        self.net.createDataset(self.normVol, self.normDayChanges)
        return

    def trainNeuralNet(self):
        self.net.train()

    def trainAndValidateNeuralNet(self):
        print('Training and validating neural net...')
        return self.net.trainAndValidate()

    def predictNeuralNet(self):
        point = self.net.createDataPoint(len(self.normVol), self.normVol, self.normDayChanges)
        return self.net.predict(point)

    def readDataFromJson(self):
        if self.volume:
            return

        filename = self.ticker + '_stockData.json'
        if path.exists(filename) != True:
            fetcher = StockDataFetcher()
            fetcher.writeDataToFile(self.ticker)
        file = open(filename, 'r')
        stockdata = json.load(file)

        for day in stockdata['chart']:
            self.dates.append(day['date'])
            self.volume.append(day['volume'])
            self.dayChanges.append(day['changePercent'])
        print(len(self.volume))
        return 

    def plotPercentChange(self):
        self.readDataFromJson()
        plt.plot(self.dates, self.dayChanges)
        plt.hlines(0, 0, len(self.dates))
        plt.xticks(rotation='vertical')
        plt.title(self.ticker + ' % change over time')
        plt.ylabel('Change in %')
        plt.xlabel('Date')
        plt.show()
    
    def plotVolume(self):
        self.readDataFromJson()
        plt.plot(self.dates, self.volume)
        plt.xticks(rotation='vertical')
        plt.title(self.ticker + ' volume over time')
        plt.ylabel('Volume')
        plt.xlabel('Date')
        plt.show()



