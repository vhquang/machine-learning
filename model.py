from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense


def make_model(units=30):
    model = Sequential()
    model.add(LSTM(
        units,
        input_shape=(4, 2))
    )
    model.add(Dense(2))
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(train_X, train_y, epochs=5, batch_size=1, verbose=2)
    return model

def main():
    pass

if __name__ == '__main__':
    main()