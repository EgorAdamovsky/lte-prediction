# %%
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, title, figure
from kan import *
import warnings
from scipy.io import loadmat
import time
import numpy as np

# matplotlib.use('gtk3agg')
warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# ОЦЕНКА ПОТЕРЬ ПРИ ОБУЧЕНИИ
def TrainMSE():
    with torch.no_grad():
        predictions = model(dataset['train_input'][0:test_data_size])
        mse = torch.nn.functional.mse_loss(predictions, dataset['train_label'][0:test_data_size])
    return mse


# ОЦЕНКА ПОТЕРЬ ПРИ ВАЛИДАЦИИ
def TestMSE():
    with torch.no_grad():
        predictions = model(dataset['test_input'][0:test_data_size])
        mse = torch.nn.functional.mse_loss(predictions, dataset['test_label'][0:test_data_size])
    return mse


# ЦВЕТА
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ОПРЕДЕЛИТЬ УСТРОЙСТВО ВЫПОЛНЕНИЯ
def SetDevice():
    if torch.cuda.is_available():
        print(Colors.GREEN + "[Инфо]" + Colors.ENDC, "рабочее устройство GPU")
        return torch.device("cuda")
    else:
        print(Colors.WARNING + "[Инфо]" + Colors.ENDC, "рабочее устройство CPU")
        return torch.device("cpu")


# ЗАГРУЗИТЬ ДАННЫЕ
def LoadData(adress, size, n, device):
    X1 = loadmat(adress + 'clear__X_test.mat')['X_test'].astype('float32')
    Y1 = loadmat(adress + 'clear__Y_test.mat')['Y_test'].astype('float32')
    X2 = loadmat(adress + 'clear__X_train.mat')['X_train'].astype('float32')
    Y2 = loadmat(adress + 'clear__Y_train.mat')['Y_train'].astype('float32')
    if n != 0:
        n = n - 1
        X2 = X2[n * size: (n + 1) * size, :]
        Y2 = Y2[n * size: (n + 1) * size, :]
        X1 = X1[int(0.2 * n * size): int(0.2 * (n + 1) * size), :]
        Y1 = Y1[int(0.2 * n * size): int(0.2 * (n + 1) * size), :]
        print("Train data start from", n * size)
    print(Colors.GREEN + "[Инфо]" + Colors.ENDC, 'данные прочитаны:')
    print('\t', '- train input ', X2.shape)
    print('\t', '- train output', Y2.shape)
    print('\t', '- test input  ', X1.shape)
    print('\t', '- test output ', Y1.shape)
    return {'train_input': torch.from_numpy(X2).to(device), 'test_input': torch.from_numpy(X1).to(device),
            'train_label': torch.from_numpy(Y2).to(device), 'test_label': torch.from_numpy(Y1).to(device)}


# %%
# ПАРАМЕТРЫ
data_size = 300000
test_data_size = 10000
layers = [5, 10, 20]
epochs = 300
lamb = 0.0001
grid_kan = 60
polyn = 3
batch = 60000
n = 1
loss = torch.nn.MSELoss()
device = SetDevice()
adr = "D:\\PSU\\orig-kan\\pykan\\my-kan\\my-kan\\try-1\\"

params = 0
for i in range(len(layers) - 1):
    params = params + layers[i] * layers[i + 1]
print('Количество параметров:', params * (grid_kan + polyn))  # 1350

# %% ОСНОВНОЙ КОД
for i in range(n):
    n_learn = i + 1  # номер итерации (до)обучения, где 1 - это первое обучение модели
    model_name = 'Input ' + str(data_size) + ', Layers ' + str(layers) + ', Lamb ' + str(
        lamb) + ', Grid ' + str(grid_kan) + ', Polyn ' + str(polyn)
    dataset = LoadData(adr, data_size, n_learn, device)

    model = KAN(width=layers, grid=grid_kan, k=polyn, device=device)
    if n_learn > 1:
        temp_name = '[' + str(n_learn - 1) + '] ' + model_name
        KAN.load_ckpt(model, name=temp_name)
        print('\n' + Colors.GREEN + "[Инфо]" + Colors.ENDC, 'загружена модель', temp_name)

    results = model.fit(dataset, metrics=(TrainMSE, TestMSE), loss_fn=loss, steps=epochs, lamb=lamb, batch=batch,
                        lr=1.0 / (i + 1))

    print(Colors.CYAN + 'MSE (train)' + Colors.ENDC, results['TrainMSE'])
    print(Colors.CYAN + 'MSE (test) ' + Colors.ENDC, results['TestMSE'])

    # ГРАФИКИ
    f1 = figure()
    ax1 = f1.add_subplot(111)
    data_len = len(results['TrainMSE'])
    ax1.plot(range(data_len), results['TestMSE'], range(data_len), results['TrainMSE'])
    title(model_name)
    ax1.grid()
    ax1.legend(('train', 'test'))

    # ПРОВЕРКА
    f2 = figure()
    ax2 = f2.add_subplot(111)
    pred = np.mean(model(dataset['train_input'][0:test_data_size]).cpu().detach().numpy() > 0.5, axis=1)
    real = np.mean(dataset['train_label'][0:test_data_size].cpu().detach().numpy(), axis=1)
    err = np.mean(abs(pred - real))
    acc = 100 * (1 - err)
    print("Accuracy (tr 0.5) = ", acc, "%")
    ax2.plot(range(len(pred)), pred, range(len(real)), real)
    title('Сравнение (ср. загруженность линий) ' + str(acc))
    ax2.legend(('model', 'test'))
    ax2.axis((0, 150, -0.2, 1.2))
    ax2.grid()

    # СОХРАНЕНИЕ
    model_name = '[' + str(n_learn) + '] ' + model_name
    model.saveckpt(model_name)
    print(Colors.GREEN + "[Инфо]" + Colors.ENDC, 'сохранена модель', model_name)

    show()

# %% ТЕСТ

print("Началось")
all_res = []
trains = []
tests = []

val_data_size = 10000
val_n = 3

for i in range(n):
    n_learn = i + 1  # номер итерации (до)обучения, где 1 - это первое обучение модели
    model_name = adr + '[' + str(n_learn) + '] ' + 'Input ' + str(data_size) + ', Layers ' + str(
        [5, 10, 20]) + ', Lamb ' + str(
        lamb) + ', Grid ' + str(grid_kan) + ', Polyn ' + str(polyn)
    print(model_name)

    dataset = LoadData(adr, val_data_size, val_n, device)

    # ОСНОВНОЙ КОД
    # model = KAN(width=layers, grid=grid_kan, k=polyn, device=device)
    model = KAN.loadckpt(path=model_name)
    with torch.no_grad():

        # ПОИСК ЛУЧШЕГО ПОРОГА
        th = [0.5]
        acc = []
        for thi in th:
            start_time = time.time()
            data = dataset['train_input']
            pred = model(data).cpu().detach().numpy()
            print("--- %s seconds ---" % (time.time() - start_time))

            # PRUNE
            # model = model.prune(node_th=0.1, edge_th=0.0000001)  # 0.04
            # start_time = time.time()
            # pred = model(dataset['train_input']).cpu().detach().numpy()
            # print("--- %s seconds ---" % (time.time() - start_time))

            for p in range(len(pred)):
                pred[p] = random.uniform(0, 1)
                pred[p] = 1

            pred = np.concatenate(
                [np.diagonal(pred[::-1, :], k)[::(2 * (k % 2) - 1)] for k in
                 range(1 - pred.shape[0], pred.shape[0])])
            pred = pred > thi
            y_test = dataset['train_label'].cpu().detach().numpy()
            y_test = np.concatenate(
                [np.diagonal(y_test[::-1, :], k)[::(2 * (k % 2) - 1)] for k in
                 range(1 - y_test.shape[0], y_test.shape[0])])
            res = 100 * (1 - np.mean(abs(pred - y_test)))
            all_res.append(res)
            acc.append(res)
            # print(thi, "-> accuracy", res, "%")
        acc = np.mean(acc)
        print(th, acc)

        plt.title('Проверка точности: ' + str(res) + "%")
        plt.plot(range(len(pred)), -1 * pred, range(len(pred)), y_test)
        plt.axis((800, 1000, -1.25, 1.25))
        plt.grid()

#
# model.plot()
plt.show()
plt.savefig("Acc.png")
print("Закончилось")
