import typing as tp

import numpy as np
import pandas as pd

PRICES_FILE = 'data/prices.csv'
ADJUST_PRICE_FILE = 'data/prices-split-adjusted.csv'


def get_stock_price(ticker: str):
    """
    Return the price data for company with given `ticker` .
    """
    df = pd.read_csv(ADJUST_PRICE_FILE)
    company = df[df['symbol'] == ticker].copy()
    company.index = pd.to_datetime(company['date'])
    company.drop(['date', 'symbol'], axis=1, inplace=True)
    return company


def get_closing_price(ticker: str, as_array=True):
    """
    Get closing price for a company with given `ticker`.
    """
    df = get_stock_price(ticker)
    series= df['close']
    if as_array:
        return np.array(series)
    return series


def make_time_windows(data: tp.Iterable, timesteps: int) -> (np.ndarray, np.ndarray):
    """
    Split the data into sequence of training inputs and expected results.
    The result is the value right after the training input.

    Args:
        timesteps: a number of timestep (interval) in each training input.

    Ex:
    >>> make_time_windows([0,1,2,3,4,5], timesteps=3)
    train: [ [0,1,2], [1,2,3] ]
    expected: [3,4]
    """
    n = len(data)
    train = [data[i: i + timesteps] for i in range(n - timesteps - 1)]
    expected = [data[i + timesteps] for i in range(n - timesteps - 1)]
    assert len(train) == len(expected)
    return np.array(train), np.array(expected)


def make_normalized_train_data(data: tp.Iterable, timesteps: int):
    """
    Normalize train and test data, base on the last known value of previous windonw.

    Ex:
    >>> make_normalized_train_data([0,1,2,3,4,5], timesteps=3)
    train: [ [-1, -0.5, 0], [-0.5, 0, 0.5] ]
    expected: [0.5, 1]
    """
    train, expected = make_time_windows(data, timesteps)
    n = len(train)
    normalizers = [train[0][-1]] + [train[i-1][-1] for i in range(1, n)]
    norm_train = [train[i] / normalizers[i] - 1 for i in range(n)]
    norm_expected = [expected[i] / normalizers[i] - 1 for i in range(n)]
    return np.array(norm_train), np.array(norm_expected), normalizers


def split(data, ratio: float=0.9):
    m = round(len(data) * ratio)
    return data[:m], data[m:]


def main():
    seq = make_normalized_train_data(np.arange(6), timesteps=3)
    print(seq[0])
    print(seq[1])

if __name__ == '__main__':
    main()