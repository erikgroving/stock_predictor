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
from StockDataFormatter import StockDataFormatter
from StockLstm import StockLstm
from sklearn.preprocessing import normalize


class StockPredictor:
    ticker = ''
    volume = []
    dayChanges = []
    dates = []
    rf = StockRandomForest(14, 4, 1000)

    def __init__(self, ticker):
        self.df = StockDataFormatter()
        self.ticker = ticker

    def createRandomForest(self, days, depth, estimators):
        self.rf = StockRandomForest(days, depth, estimators)
        self.readDataFromJson()
        self.rf.createDataset(self.volume[1:], self.dayChanges[1:])
    
    def trainRandomForest(self):
        self.rf.train()

    def trainAndValidateRandomForest(self):
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

    def normalizeData(self):
        self.normVol = torch.Tensor(np.asarray(normalize([self.volume[1:]])[0]))
        self.normDayChanges = torch.Tensor(np.asarray(normalize([self.dayChanges[1:]])[0]))

    def createLstm(self, input_size, hidden_size, num_layers):
        self.readDataFromJson()
        self.normalizeData()
        data, labels = self.df.createDataset(self.normVol, self.dayChanges)
        self.lstm = StockLstm(input_size, hidden_size, num_layers)
        self.lstm.setDataAndLabels(data, labels)
        return

    def trainLstm(self):
        self.lstm.train()

    def predictLstm(self):
        self.lstm.predict()


    def createNeuralNet(self, days):
        self.readDataFromJson()
        self.normalizeData()
        print(self.normVol.shape)
        data, labels = self.df.createDataset(self.normVol, self.dayChanges)
        self.net = StockNeuralNet(data, labels, days, 30)
        return

    def createNeuralNetWithSetAndLabels(self, data, labels, days):
        self.net = StockNeuralNet(data, labels, days, 20)
        return


    def trainNeuralNet(self):
        self.net.train()

    def trainAndValidateNeuralNet(self):
        return self.net.trainAndValidate()

    def predictNeuralNet(self):
        point = self.df.createDataPoint(len(self.normVol), self.normVol, self.normDayChanges)
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
        return 

    def generateBacktestSets(self, daysForTraining, daysPerInput):
        self.readDataFromJson()
        self.normalizeData()
        self.formatter = StockDataFormatter(daysForTraining, daysPerInput)
        self.sets, self.testPoints = self.formatter.genBacktestingDatasets(self.volume, self.dayChanges)

    def backtestNeuralNet(self, daysForTraining, daysPerInput):
        self.generateBacktestSets(daysForTraining, daysPerInput)
        totalPoints = 0
        numCorrect = 0
        for i in range(len(self.sets)):
            data = self.sets[i][0]
            labels = self.sets[i][1]
            testdata = self.testPoints[i][0]
            testlabel = self.testPoints[i][1]
            self.createNeuralNetWithSetAndLabels(data, labels, daysPerInput)
            self.trainNeuralNet()
            pred = self.net.predict(testdata)
            if torch.argmax(pred) == testlabel:
                numCorrect += 1
            totalPoints += 1
        print(numCorrect / totalPoints)
        return numCorrect / totalPoints

    def backtestLstm(self, input_size, hidden_size, num_layers, daysForTraining, daysPerInput):
        self.generateBacktestSets(daysForTraining, daysPerInput)
        totalPoints = 0
        numCorrect = 0
        for i in range(len(self.sets)):
            data = self.sets[i][0]
            data = data.reshape(-1, daysPerInput, input_size)

            labels = self.sets[i][1]
            testdata = self.testPoints[i][0].reshape(1, daysPerInput, input_size).
            testlabel = self.testPoints[i][1]
            self.createLstm(input_size, hidden_size, num_layers)
            self.lstm.setDataAndLabels(data, labels)
            self.trainLstm()
            pred = self.lstm.predict(testdata)
            if torch.argmax(pred) == testlabel:
                numCorrect += 1
            totalPoints += 1
        print(numCorrect / totalPoints)
        return numCorrect / totalPoints


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



