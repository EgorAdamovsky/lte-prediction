import kan
from scipy.io import loadmat


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
    if kan.torch.cuda.is_available():
        print(Colors.GREEN + "[Инфо]" + Colors.ENDC, "рабочее устройство GPU")
        return kan.torch.device("cuda")
    else:
        print(Colors.WARNING + "[Инфо]" + Colors.ENDC, "рабочее устройство CPU")
        return kan.torch.device("cpu")


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
    print(Colors.GREEN + "[Инфо]" + Colors.ENDC, 'данные прочитаны:')
    print('\t', '- train input ', X2.shape)
    print('\t', '- train output', Y2.shape)
    print('\t', '- test input  ', X1.shape)
    print('\t', '- test output ', Y1.shape)
    return {'train_input': kan.torch.from_numpy(X2).to(device),
            'test_input': kan.torch.from_numpy(X1).to(device),
            'train_label': kan.torch.from_numpy(Y2).to(device),
            'test_label': kan.torch.from_numpy(Y1).to(device)}
