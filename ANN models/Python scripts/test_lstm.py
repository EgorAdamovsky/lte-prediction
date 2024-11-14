import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from info import *
from keras.src.saving.saving_api import load_model
from numpy import mean
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ПОДГОТОВКА
adress = "..\\clear-small-data\\"
X_test, Y_test, _, _ = LoadData(adress)  # загрузка данных

model = load_model('bigmodel-small-part-data-lstm-2.keras')  # загрузка модели

# ПРОГНОЗ
Y_pred = model.predict(X_test)  # прогноз модели

# ИНТЕР-ПОДГОТОВКА
L = len(Y_test)  # полная длина данных
Lr = range(0, L)

# ПОИСК ЛУЧШЕГО ПОРОГА
th = np.arange(0.05, 0.95, 0.05)
acc = []
for i in range(0, len(th)):
    Y_pred_bin = Y_pred > th[i]
    err = sum(sum(abs(Y_pred_bin - Y_pred))) / (20 * L)
    acc.append(err)
    print("{0:.2f}".format(th[i]), " -> ", "{0:.5f}".format(acc[i]))
best_th = th[acc.index(min(acc))]
print("Лучший порог: ", "{0:.2f}".format(best_th))

Y_pred_bin = Y_pred > best_th
Y_pred_mean_bin = mean(Y_pred_bin, axis=1)  # усреднение за кадр проверочных данных
Y_test_mean = mean(Y_test, axis=1)  # усреднение за кадр тестовых данных

# plt.figure(1)
# plt.plot(Lr, Y_test_mean, Lr, -Y_pred_mean_bin)
# plt.axis([0, 200, -1.1, 1.1])
# plt.grid(visible=True)
# plt.show()

res = 1 - abs(Y_test_mean - Y_pred_mean_bin)
print(mean(res))

plt.figure(2)
plt.plot(Lr, Y_test_mean,)
plt.axis([0, 200, 0, 1.1])
plt.grid(visible=True)
plt.show()

#
# plt.figure(2)
# for i in range(0, 10):
#     plt.subplot(10, 1, i + 1)
#     plt.plot(range(0, 20), Y_test[i, :], range(0, 20), Y_pred[i, :])
#     plt.axis([0, 20, 0, 1.25])

