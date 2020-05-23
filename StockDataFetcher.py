from StockDataApi import StockDataApi
import json

class StockDataFetcher:
    def writeDataToFile(self, ticker):
        api = StockDataApi()
        params = []
        params.append(("types", "chart"))
        params.append(("range", "3m"))
        params.append(("last", "100"))
        result = api.getTickerPriceAndVolumeData(ticker, params)

        filename = ticker + '_stockData.json'

        f = open(filename, 'w')
        f.write(json.dumps(result, indent=4))
        f.close()
        print(json.dumps(result, indent=4))

