import pickle
import socket

import joblib
import numpy as np
from termcolor import cprint

from conf import ROUNDS, FRAC, GAR, LEARNING_RATE, BLOCK_SIZE, F, ATTACK


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


def log(mtype, message):
    if mtype == "result":
        cprint(" Result:  ", 'blue', attrs=['reverse'], end=' ')
        cprint(message, 'blue')
    elif mtype == "error":
        cprint(" Error:   ", 'red', attrs=['reverse'], end=' ')
        cprint(message, 'red')
    elif mtype == "success":
        cprint(" Success: ", 'green', attrs=['reverse'], end=' ')
        cprint(message, 'green')
    elif mtype == "event":
        cprint(" Event:   ", 'cyan', attrs=['reverse'], end=' ')
        cprint(message, 'cyan')
    elif mtype == "warning":
        cprint(" Warning: ", 'yellow', attrs=['reverse'], end=' ')
        cprint(message, 'yellow')
    elif mtype == "info":
        cprint(" Info:    ", attrs=['reverse'], end=' ')
        cprint(message)
    else:
        cprint(" Log:     ", 'magenta', attrs=['reverse'], end=' ')
        cprint(message, 'magenta')


def create_tcp_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, TCP_SOCKET_BUFFER_SIZE)
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, TCP_SOCKET_BUFFER_SIZE)
    # sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def load_params(params):
    defaults = Map(
        {'rounds': ROUNDS, 'frac': FRAC, 'lr': LEARNING_RATE, 'gar': GAR, 'f': F, 'attack': ATTACK,
         'block_size': BLOCK_SIZE})
    if isinstance(params, Map):
        return Map(dict(defaults, **params))
    elif params is not None:
        log('warning', "The variable params should be an instance of `Map")
    else:
        return defaults


def input_size(model: str, dataset: str):
    if model.upper() in ["LR", "LN", "SVM"]:
        if dataset.lower() == "mnist":
            return 784
        elif dataset.lower() == "boston":
            return 14
        elif dataset.lower() == "phishing":
            return 68
        else:
            log('error', f"Unknown dataset {dataset}")
            exit(0)
    else:
        if dataset.lower() == "mnist":
            return [784, 30, 10]
        else:
            log('error', f"Unsupported or Unknown dataset {dataset}")
            exit(0)


def chunks(l, n):
    if n > 0:
        np.random.shuffle(l)
        output = [l[i:i + n] for i in range(0, len(l), n)]
        s = len(output[-1])
        if s < n:
            output[-1].extend(output[-2][s:n])
        return output
    else:
        return None


def mnist(path, binary=True):
    try:
        open(path, 'r')
    except FileNotFoundError as e:
        log('error', str(e))
        exit()
    X_train, Y_train = joblib.load(path)
    Y_train = Y_train.astype(int).reshape(-1, 1)
    if binary:
        # Extract 1 and 2 from train dataset
        f1 = 1
        f2 = 2
        Y_train = np.squeeze(Y_train)
        X_train = X_train[np.any([Y_train == f1, Y_train == f2], axis=0)]
        Y_train = Y_train[np.any([Y_train == f1, Y_train == f2], axis=0)]
        Y_train = Y_train - f1
        Y_train = Y_train.reshape(-1, 1)
    else:
        Y_train = np.array([np.eye(1, 10, k=int(y)).reshape(10) for y in Y_train])
    X_train = X_train / 255

    return X_train, Y_train


def load_test_dataset(dataset):
    if dataset == "mnist":
        return mnist(path='./datasets/mnist.data')


def save(filename, data):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)
        print("Writing to file", filename)
    return
