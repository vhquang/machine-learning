from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
# import numpy as np

from data_module import get_closing_price, split

ticker = 'GOOGL'
test_ratio = 0.9
window_size = 4


def arima_forecast():
    data = get_closing_price(ticker, as_array=True)
    train, test = split(data, ratio=test_ratio)
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history,
                      order=(window_size, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        # print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    # print('Test MSE: %.3f' % error)
    # plot
    return predictions, test


def main():
    arima_forecast()

if __name__ == '__main__':
    main()
