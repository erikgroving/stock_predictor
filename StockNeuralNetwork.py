# Inputs:
# Volume for the day
# Percent change for the day

# Outputs:
# Whether or not the next day is going to green or red, using softmax

# Hyperparameters:
# Number of days to use

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split

class NeuralNet(nn.Module):

    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
         
class StockNeuralNet:
    def __init__(self, n_previousDays, epochs):
        self.n_previousDays = n_previousDays
        self.net = NeuralNet(n_previousDays * 2)
        self.net.double()
        #self.net.cuda()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.net.parameters(), 1e-2)
        self.epochs = epochs

        
    def createDataset(self, volume, dayChanges):
        numberOfDays = len(volume)
        numPoints = len(range(self.n_previousDays, numberOfDays))

        self.data = torch.zeros(numPoints, self.n_previousDays * 2).double()
        self.labels = torch.zeros(numPoints).long()

        for i in range(self.n_previousDays, numberOfDays):
            self.data[i - self.n_previousDays] = self.createDataPoint(i, volume, dayChanges)
            self.labels[i - self.n_previousDays] = self.createLabelPoint(dayChanges[i])
        
        print(self.data)
        print(self.labels)
        print(self.data.size())
        print(self.labels.size())
    
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


    def train(self):
        for j in range(self.epochs):
            y_pred = self.net(self.data)
            loss = self.criterion(y_pred, self.labels)
            print(j, loss.item())

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
    
    def trainWithData(self, data, labels, test_data, test_labels):
        for j in range(self.epochs):
            y_pred = self.net(data)
            loss = self.criterion(y_pred, labels)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def trainAndValidate(self):
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.1)
        self.trainWithData(x_train, y_train, x_test, y_test)
        preds = self.predict(x_test)
        
        print("Total accuracy: " + str(self.printAccuracy(preds, y_test)))
        return self.printAccuracy(preds, y_test)

    
    def printAccuracy(self, preds, labels):
        num_correct = 0
        for i in range(len(preds)):
            if torch.argmax(preds[i]) == labels[i]:
                num_correct += 1

        accuracy = num_correct / len(labels)
        return accuracy


    def predict(self, input):
        return self.net(input.double())