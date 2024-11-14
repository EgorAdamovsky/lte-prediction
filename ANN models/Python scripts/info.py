import scipy
from matplotlib import pyplot as plt
from numpy import mean


# Загрузка данных
def LoadData(adress):
    X_test = scipy.io.loadmat(adress + 'clear__X_test.mat')['X_test'].astype('float32')
    Y_test = scipy.io.loadmat(adress + 'clear__Y_test.mat')['Y_test'].astype('float32')
    X_train = scipy.io.loadmat(adress + 'clear__X_train.mat')['X_train'].astype('float32')
    Y_train = scipy.io.loadmat(adress + 'clear__Y_train.mat')['Y_train'].astype('float32')
    print('Данные прочитаны: ' + adress)
    return X_test, Y_test, X_train, Y_train


# Отображение данных
def ShowData(X_train, Y_train):
    plt.figure(0)
    plt.subplot(6, 1, 1)
    plt.plot(range(0, len(Y_train)), mean(Y_train, axis=1))
    # plt.plot(range(0, len(Y_train)), Y_train)
    plt.axis([0, 100, 0, 1.25])
    for i in range(0, 5):
        plt.subplot(6, 1, i + 2)
        plt.plot(range(0, len(X_train)), X_train[:, i])
        plt.axis([0, 100, 0, 1.25])
    plt.show()


# Получение обычного списка из тензора Трансформера
def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list
