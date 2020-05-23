# Inputs:
# Volume for the day
# Percent change for the day

# Outputs:
# Whether or not the next day is green, binary, yes or no

# Hyperparameters:
# Number of days to use

from sklearn.ensemble import RandomForestClassifier

class StockRandomForest:

    data = []
    labels = []
    randomForestClassifier = RandomForestClassifier()

    def __init__(self, depth, estimators):
        self.randomForestClassifier = RandomForestClassifier(max_depth=depth, n_estimators=estimators)

    def setData(self, data):
        self.data = data
    
    def setLabels(self, labels):
        self.labels = labels

    def train(self):
        self.randomForestClassifier.fit(self.data, self.labels)

    def predict(self, input):
        return self.randomForestClassifier.predict(input)
