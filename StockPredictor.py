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

    def __init__(self, ticker):
        self.ticker = ticker
        self.volume = []
        self.dayChanges = []
        self.dates = []
        self.close = []
        self.rf = StockRandomForest(14, 4, 1000)

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
        return output

    def normalizeData(self):
        self.normVol = torch.Tensor(np.asarray(normalize([self.volume[1:]])[0]))
        self.normDayChanges = torch.Tensor(np.asarray(normalize([self.dayChanges[1:]])[0]))

    def createLstm(self, input_size, hidden_size, num_layers, daysForTrain, sequence_length):
        self.readDataFromJson()
        self.normalizeData()
        self.df = StockDataFormatter(daysForTrain, sequence_length)
        data, labels = self.df.createDataset(self.normVol, self.dayChanges)
        data = data.reshape(-1, sequence_length, 2)
        self.lstm = StockLstm(input_size, hidden_size, num_layers)
        self.lstm.setDataAndLabels(data, labels)
        return

    def trainLstm(self):
        self.lstm.train()

    def predictLstm(self):
        point = self.df.createDataPoint(len(self.normVol), self.normVol, self.normDayChanges)
        point = point.reshape(1, -1, 2)
        return torch.argmax(self.lstm.predict(point)).item()


    def createNeuralNet(self, daysForTraining, daysForInput):
        self.readDataFromJson()
        self.normalizeData()
        self.df = StockDataFormatter(daysForTraining, daysForInput)
        data, labels = self.df.createDataset(self.normVol, self.dayChanges)
        self.net = StockNeuralNet(data, labels, daysForInput, 20)

    def createNeuralNetWithSetAndLabels(self, data, labels, days):
        self.net = StockNeuralNet(data, labels, days, 20)
        return


    def trainNeuralNet(self):
        self.net.train()

    def trainAndValidateNeuralNet(self):
        return self.net.trainAndValidate()

    def predictNeuralNet(self):
        point = self.df.createDataPoint(len(self.normVol), self.normVol, self.normDayChanges)
        return torch.argmax(self.net.predict(point)).item()

    def addVolumeAndChangeToData(self, volumePoint, dayChangePoint):
        self.readDataFromJson()
        self.volume.append(volumePoint)
        self.dayChanges.append(dayChangePoint)

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
            self.close.append(day['close'])
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
        numGreenPreds = 0
        numGreenPredAndDays = 0
        numRedPreds = 0
        numRedPredAndDays = 0
        numGreenDays = 0
        numRedDays = 0
        for i in range(len(self.sets)):
            data = self.sets[i][0]
            data = data.reshape(-1, daysPerInput, input_size)

            labels = self.sets[i][1]
            testdata = self.testPoints[i][0].reshape(1, daysPerInput, input_size)
            testlabel = self.testPoints[i][1]
            self.createLstm(input_size, hidden_size, num_layers)
            self.lstm.setDataAndLabels(data, labels)
            self.trainLstm()
            pred = self.lstm.predict(testdata)
            if torch.argmax(pred) == testlabel:
                numCorrect += 1

            if torch.argmax(pred).item() == 1:
                numGreenPreds += 1
                if testlabel.item() == 1:
                    numGreenPredAndDays += 1
            
            if torch.argmax(pred).item() == 0:
                numRedPreds += 1
                if testlabel.item() == 0:
                    numRedPredAndDays += 1

            if testlabel.item() == 0:
                numRedDays += 1
            else:
                numGreenDays += 1

            totalPoints += 1

        #print('Red days: ', str(numRedDays))
        #print('Green days: ', numGreenDays)

        greenPredAcc =  numGreenPredAndDays / numGreenPreds
        redPredAcc = numRedPredAndDays / numRedPreds
        overallAcc = numCorrect / totalPoints
        return greenPredAcc, redPredAcc, overallAcc

    def getPercentChange(self):
        self.readDataFromJson()
        return self.dates, self.dayChanges, self.close

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



