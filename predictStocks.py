from StockPredictor import StockPredictor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import sys

n_days = 8

# random forest parameters
depth = 8
estimators = 1000

#LSTM params
input_size = 2 # volume + % change
hidden_size = 24
sequence_length = 8
num_layers = 2
daysForTraining = 20

def backtestLstm(ticker):
    sum = 0
    n_tests = 10

    predictor = StockPredictor(ticker)
    greenPredAcc = 0
    redPredAcc = 0
    overallAcc = 0
    for i in range(n_tests):
        g_acc, r_acc, o_acc = predictor.backtestLstm(input_size, hidden_size, num_layers, daysForTraining, sequence_length) 
        print('Green Prediction Accuracy: ', str(g_acc))
        print('Red Prediction Accuracy: ', str(r_acc))
        print('Overall Accuracy: ', str(o_acc), '\n\n')

        greenPredAcc += g_acc
        redPredAcc += r_acc
        overallAcc += o_acc
        
    greenPredAcc /= n_tests
    redPredAcc /= n_tests
    overallAcc /= n_tests
    print('Final Results')
    print('Green Prediction Accuracy: ', str(greenPredAcc))
    print('Red Prediction Accuracy: ', str(redPredAcc))
    print('Overall Accuracy: ', str(overallAcc))

def predictTickerLstm(ticker, addTodaysData = False, vol = 0, chg = 0):
    predictor = StockPredictor(ticker)
    if addTodaysData:
        predictor.addVolumeAndChangeToData(vol, chg)
    predictor.createLstm(input_size, hidden_size, num_layers, daysForTraining, sequence_length)
    predictor.trainLstm()
    pred = predictor.predictLstm()
    return pred

def runAggregateLstmPred(ticker, addTodaysData = False, vol = 0, chg = 0):
    greenPreds = 0
    redPreds = 0
    num_preds = 500
    for i in range(num_preds):
        pred = predictTickerLstm(ticker, addTodaysData, vol, chg)
        
        if i % 200 == 0:
            print(i, '/', num_preds)
        
        if pred == 1:
            greenPreds += 1
        else:
            redPreds += 1
    
    if greenPreds > redPreds:
        print('LSTM predicts that ', ticker, ' will be green tomorrow')
    else:
        print('LSTM predicts that ', ticker, ' will be red tomorrow')
    print('Green predictions: ', greenPreds)
    print('Red predictions: ', redPreds)

def predictTickerNeuralNet(ticker):
    predictor = StockPredictor(ticker)
    predictor.createNeuralNet(daysForTraining, n_days)
    predictor.trainNeuralNet()
    return predictor.predictNeuralNet()

def runAggregateNnPred(ticker):
    greenPreds = 0
    redPreds = 0
    num_preds = 1000
    for i in range(num_preds):
        pred = predictTickerNeuralNet(ticker)
       
        if i % 200 == 0:
            print(i, '/', num_preds)

        if pred == 1:
            greenPreds += 1
        else:
            redPreds += 1
    
    if greenPreds > redPreds:
        print('Neural net predicts that ', ticker, ' will be green tomorrow')
    else:
        print('Neural net predicts that ', ticker, ' will be red tomorrow')
    print('Green predictions: ', greenPreds)
    print('Red predictions: ', redPreds)

def predictTickerRandomForest(ticker):
    predictor = StockPredictor(ticker)
    predictor.createRandomForest(n_days, depth, estimators)
    predictor.trainRandomForest()
    return predictor.predictRandomForest()

def runAggregateRandomForestPred(ticker):
    greenPreds = 0
    redPreds = 0
    num_preds = 20
    for i in range(num_preds):
        pred = predictTickerRandomForest(ticker)
       
        if i % 20 == 0:
            print(i, '/', num_preds)

        if pred == 1:
            greenPreds += 1
        else:
            redPreds += 1
    
    if greenPreds > redPreds:
        print('Random forest predicts that ', ticker, ' will be green tomorrow')
    else:
        print('Random forest predicts that ', ticker, ' will be red tomorrow')
    print('Green predictions: ', greenPreds)
    print('Red predictions: ', redPreds)

def runAllAggregates(ticker):
    print('Running random forest ensemble model...')
    runAggregateRandomForestPred(ticker)
    print('Running neural network ensemble model...')
    runAggregateNnPred(ticker)
    print('Running LSTM ensemble model...')
    runAggregateLstmPred(ticker)

for arg in sys.argv[1:]:
    predictor = StockPredictor(arg)
    dates, changes = predictor.getPercentChange()
    plt.plot(dates, changes, label=arg)

plt.hlines(0, 0, len(dates))
plt.xticks(rotation='vertical')
plt.title('% change over time')
plt.ylabel('Change in %')
plt.xlabel('Date')
plt.legend()
plt.show()