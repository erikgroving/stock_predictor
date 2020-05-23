import json
import matplotlib.pyplot as plt
from os import path
from StockDataApi import StockDataApi
from StockDataFetcher import StockDataFetcher

class StockPredictor:
    def readDataFromJson(self, ticker):
        filename = ticker + '_stockData.json'
        if path.exists(filename) != True:
            fetcher = StockDataFetcher()
            fetcher.writeDataToFile(ticker)
        file = open(filename, 'r')
        stockdata = json.load(file)

        dates = []
        dayChanges = []
        for day in stockdata['chart']:
            dates.append(day['date'])
            dayChanges.append(day['changePercent'])
        return dates, dayChanges

    def plotPercentChange(self, ticker):
        dates, dayChanges = self.readDataFromJson(ticker)
        plt.plot(dates, dayChanges)
        plt.hlines(0, 0, len(dates))
        plt.xticks(rotation='vertical')
        plt.title(ticker + ' % change over time')
        plt.ylabel('Change in %')
        plt.xlabel('Date')
        plt.show()


