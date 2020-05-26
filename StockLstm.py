import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from sklearn.model_selection import train_test_split

class Lstm(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(Lstm, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).double()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).double()

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class StockLstm:

    def __init__(self, input_size, hidden_size, num_layers, epochs = 10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.net= Lstm(input_size, hidden_size, num_layers)
        self.net.double()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.net.parameters(), 1e-3)
    
    def setDataAndLabels(self, data, labels):
        self.data = data
        self.labels = labels

    def train(self):
        for j in range(self.epochs):
            y_pred = self.net(self.data.cuda())
            loss = self.criterion(y_pred, self.labels)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
    
    def trainWithData(self, data, labels, test_data, test_labels):
        for j in range(self.epochs):
            y_pred = self.net(data.cuda())
            loss = self.criterion(y_pred, labels)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
    
    def printAccuracy(self, preds, labels):
        num_correct = 0
        for i in range(len(preds)):
            if torch.argmax(preds[i]) == labels[i]:
                num_correct += 1

        accuracy = num_correct / len(labels)
        return accuracy


    def predict(self, input):
        return self.net(input.double().cuda())