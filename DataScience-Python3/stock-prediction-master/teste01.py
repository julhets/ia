import time
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd

import sklearn.preprocessing as prep

df = pd.read_csv('iris.csv')
print(df.head())


def preprocess_data(dataFrame):
    data = dataFrame.values

    result = np.array(data)

    result = np.take(result, np.random.permutation(result.shape[0]), axis=0, out=result);

    row = round(0.9 * result.shape[0])
    x_train = result[: int(row), :-1]
    y_train = result[: int(row), -1]

    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    #x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    #x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    return [x_train, y_train, x_test, y_test]


def build_model(layers):
    model = Sequential()

    # By setting return_sequences to True we are able to stack another LSTM layer
    # model.add(LSTM(
    #     input_dim=layers[0],
    #     output_dim=layers[1],
    #     return_sequences=True))
    # model.add(Dropout(0.4))
    #
    # model.add(LSTM(
    #     layers[2],
    #     return_sequences=False))
    # model.add(Dropout(0.3))
    #
    # model.add(Dense(
    #     output_dim=layers[2]))
    # model.add(Activation("linear"))
    #
    # start = time.time()
    # model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    # print("Compilation Time : ", time.time() - start)


    model.add(Dense(512, activation='relu', input_shape=(4,)))
    model.add(Dense(1, activation='softmax'))

    return model


x_train, y_train, x_test, y_test = preprocess_data(df[:: -1])
print("X_train", x_train.shape)
print("y_train", y_train.shape)
print("X_test", x_test.shape)
print("y_test", y_test.shape)

input_dim = 1
output_dim = 1
lstm_param = 4
dense_output_dim = 1
model = build_model([input_dim, output_dim, lstm_param, dense_output_dim])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# model.fit(
#     x_train,
#     y_train,
#     batch_size=151,
#     epochs=5,
#     validation_split=0.1,
#     verbose=0)

model.fit(
    x_train,
    y_train,
    batch_size=151,
    epochs=10,
    verbose=2,
    validation_data=(x_test, y_test)
)

# trainScore = model.evaluate(x_train, y_train, verbose=0)
# print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
#
# testScore = model.evaluate(x_test, y_test, verbose=0)
# print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

tmp = np.array([[[5.1,3.5,1.4,0.2]]])
pred = model.predict(tmp)
print(pred)