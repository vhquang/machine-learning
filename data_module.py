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

def get_closing_price(ticker: str, drop_date_index=True):
    """
    Get closing price for a company with given `ticker`.
    """
    df = get_closing_price(ticker)
    series= df['close']
    if drop_date_index:
        return np.array(df)
    return series

def split(data, ratio: float=0.9):
    # TODO maybe we can do this after prepare data,
    # to prevent losing data point while rouding
    m = round(len(data) * ratio)
    return data[:m], data[m:]

def make_time_windows(data: tp.Iterable, timesteps: int, window_size: int) -> (np.ndarray, np.ndarray):
    """
    Split the data into sequence of training inputs and expected results.
    The result is the value right after the training input.

    Args:
        timesteps: a number of timestep (interval) in each training input.
        window_size: a length of each timestep

    Ex:
    >>> make_time_windows([1,2,3,4,5,6], timesteps=3, window_size=2)
    """
    seq = [np.array(data[window_size * i: window_size * (i+1)])
          for i in range(len(data) // window_size)]
    train = [seq[i: i + timesteps] for i in range(len(seq) - timesteps)]
    check = [seq[i + timesteps] for i in range(len(seq) - timesteps)]
    assert len(train) == len(check)
    return np.array(train), np.array(check)


def main():
    seq = make_time_windows([1,2,3,4,5,6], timesteps=3, window_size=2)
    print(seq[0], seq[1])
    print()

if __name__ == '__main__':
    main()