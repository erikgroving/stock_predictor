# Formats data into sets that will allow for the data to be backtested
import torch
import numpy as np

class StockDataFormatter:

    def __init__(self, daysForTraining=50, daysPerInput=14):
        self.daysForTraining = daysForTraining
        self.n_previousDays = daysPerInput

    def genBacktestingDatasets(self, volume, daychanges):
        startPt = 0
        endPt = self.daysForTraining
        sets = []
        testPoints = []
        while endPt != len(volume):
            sets.append(self.createDataset(volume[startPt : endPt], daychanges[startPt : endPt]))
            testPoints.append((self.createDataPoint(endPt, volume, daychanges), self.createLabelPoint(daychanges[endPt])))
            startPt += 1
            endPt += 1

        return sets, testPoints

    def createDataset(self, volume, dayChanges):
        numberOfDays = len(volume)
        numPoints = len(range(self.n_previousDays, numberOfDays))

        data = torch.zeros(numPoints, self.n_previousDays * 2).double()
        labels = torch.zeros(numPoints).long()

        for i in range(self.n_previousDays, numberOfDays):
            data[i - self.n_previousDays] = self.createDataPoint(i, volume, dayChanges)
            labels[i - self.n_previousDays] = self.createLabelPoint(dayChanges[i])

        return data, labels
        
    def createDataPoint(self, idx, volume, dayChanges):
        startPt = idx - self.n_previousDays

        points = torch.Tensor(np.zeros(self.n_previousDays * 2))
        for i in range(startPt, idx):
            points[i - startPt] = volume[i]
        for i in range(idx - self.n_previousDays, idx):
            points[i - startPt + self.n_previousDays] = dayChanges[i]
        return points

    def createLabelPoint(self, change):
        if change > 0:
            return torch.Tensor([1]).long()
        else:
            return torch.Tensor([0]).long()

