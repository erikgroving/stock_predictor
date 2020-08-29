import requests

# RSI
# Past 14 days of open/close/%
# Past 3 days on the 5 minute chart?
# Volume?


# Steps
# Generate training and testing data by using a stocker market API for some ticker
# Train a classifier (random forest, LSTM) on the data. LSTM will use the past 14 days of data
# Generate output prediction for green or red on the next day

class StockDataApi:
    url_base = 'https://cloud.iexapis.com/'
    ver = 'stable/'
    token = 'pk_64987d421e994d87b1b0b096ab7db3e6'

    def performQuery(self, query):
        r = requests.get(query)
        return r.json()

    def getTickerPriceAndVolumeData(self, ticker, params):
        query = self.createBaseQueryString(ticker)
        query += '/batch/'
        query += self.createParamString(params)
        print(query)
        return self.performQuery(query)
        
    def getRsiData(self, ticker, range, period):
        params = []
        params.append(('range', range))
        params.append(('period', period))
        query = self.createBaseQueryString(ticker) + '/indicator/rsi/'
        query += self.createParamString(params)
        print(query)
        return self.performQuery(query)

    def createBaseQueryString(self, ticker):
        return self.url_base + self.ver + 'stock/' + ticker 
                

    def createParamString(self, params):
        firstParam = True
        paramString = '?'
        params.insert(0, ('token', self.token))
        for [label, value] in params:
            if firstParam != True:
                paramString += '&'
            firstParam = False
            
            paramString += label + '=' + value
        return paramString

