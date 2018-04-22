from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense
import numpy as np


def make_model(input_shape, units=30):
    model = Sequential()
    model.add(LSTM(
        units,
        input_shape=input_shape)
    )
    model.add(Dense(input_shape[-1]))
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(train_X, train_y, epochs=5, batch_size=1, verbose=2)
    return model


def naive_prediction(timesteps_data):
    """Predict the next price just by averaging previous window"""
    res = [x.mean() for x in timesteps_data]
    return np.array(res)


def main():
    from data_module import get_closing_price, make_time_windows
    price_array = get_closing_price('GOOGL', as_array=True)
    train, test = make_time_windows(price_array, timesteps=4)
    res = naive_prediction(train)

if __name__ == '__main__':
    main()