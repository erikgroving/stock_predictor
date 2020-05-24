# Inputs:
# Volume for the day
# Percent change for the day

# Outputs:
# Whether or not the next day is green, binary, yes or no

# Hyperparameters:
# Number of days to use

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

class StockRandomForest:
    
    n_previousDays = 14
    data = []
    labels = []
    randomForestClassifier = RandomForestClassifier()

    def __init__(self, n_previousDays, depth, estimators):
        self.n_previousDays = n_previousDays
        self.randomForestClassifier = RandomForestClassifier(max_depth=depth, n_estimators=estimators)

    def createDataset(self, volume, dayChanges):
        numberOfDays = len(volume)
        for i in range(self.n_previousDays, numberOfDays):
            self.data.append(np.asarray(self.createDataPoint(i, volume, dayChanges)))
            self.labels.append(self.createLabelPoint(dayChanges[i]))
    
    def createDataPoint(self, idx, volume, dayChanges):
        points = []
        for i in range(idx - self.n_previousDays, idx):
            points.append(volume[i])
        for i in range(idx - self.n_previousDays, idx):
            points.append(dayChanges[i])
        return points

    def createLabelPoint(self, change):
        if change > 0:
            return 1
        else:
            return 0

    def train(self):
        self.randomForestClassifier.fit(self.data, self.labels)

    def trainWithData(self, data, labels):
        self.randomForestClassifier.fit(data, labels)

    def trainAndValidate(self):
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.1)
        self.trainWithData(x_train, y_train)
        preds = self.predict(x_test)

        num_correct = 0.
        for i in range(len(preds)):
            if preds[i] == y_test[i]:
                num_correct += 1.
        accuracy = num_correct / len(y_test)
        return accuracy



    def predict(self, input):
        return self.randomForestClassifier.predict(input)