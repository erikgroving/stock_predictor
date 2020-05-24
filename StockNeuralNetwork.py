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
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
         
class StockNeuralNet:
    def __init__(self, data, labels, n_previousDays, epochs):
        self.n_previousDays = n_previousDays
        self.net = NeuralNet(n_previousDays * 2)
        self.net.double()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.net.parameters(), 5e-3)
        self.epochs = epochs
        self.data = data
        self.labels = labels

        
    def train(self):
        for j in range(self.epochs):
            y_pred = self.net(self.data)
            loss = self.criterion(y_pred, self.labels)

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