from keras.layers import LSTM
from numpy import reshape

from info import *
from keras import *
from keras.optimizers import Adam
from keras.src.layers import Dense, Dropout, Activation
from tensorflow.python.client import device_lib
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
print(device_lib.list_local_devices())

# ПОДГОТОВКА ДАННЫХ
adress = "..\\clear-small-data\\"
X_test, Y_test, X_train, Y_train = LoadData(adress)  # загрузка данных
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1], 1)
# X_train = np.array([X_train.transpose()])
# X_train = X_train.transpose()
# Y_train = np.array([Y_train.transpose()])
# Y_train = Y_train.transpose()
# ShowData(X_train, Y_train)  # отображение данных

# ПАРАМЕТРЫ
drop = 0.1
ep = 300
lays = [5, 57, 20]
batch = None

# МОДЕЛЬ
model = Sequential()
model.add(LSTM(5, return_sequences=True, input_shape=(5, 1)))
model.add(Dropout(drop))
# model.add(LSTM(lays[1], return_sequences=True))
# model.add(Dropout(drop))
# model.add(LSTM(lays[2], return_sequences=True))
# model.add(Dropout(drop))
# model.add(LSTM(lays[3], return_sequences=True))
# model.add(Dropout(drop))
# model.add(LSTM(lays[4]))
model.add(LSTM(lays[1]))
model.add(Dense(20, activation="sigmoid"))
model.compile(loss='mse', optimizer=Adam(), metrics=['mse'])
model.build()
model.summary()

# ОБУЧЕНИЕ
net = model.fit(X_train, Y_train, epochs=ep, batch_size=batch, )
model.save('bigmodel-small-part-data-lstm-3.keras')
plt.plot(net.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# ПРОВЕРКА
results = model.evaluate(X_test, Y_test)
print('test loss, test acc:', results)
print('ok')
