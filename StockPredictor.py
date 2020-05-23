import json
import matplotlib.pyplot as plt
from os import path
from StockDataApi import StockDataApi
from StockDataFetcher import StockDataFetcher

class StockPredictor:
    ticker = ''
    volume = []
    dayChanges = []
    dates = []

    def __init__(self, ticker):
        self.ticker = ticker

    def createRandomForest(self):
        return

    def createLstm(self):
        return

    def createNeuralNet(self):
        return

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



