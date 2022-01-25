import pickle

import numpy as np

from .utils import log

DISCONNECT = 0
TRAIN_JOIN = 1
TRAIN_START = 2
TRAIN_INFO = 3
TRAIN_STOP = 5


def join_train(model, params):
    if model is None:
        exit(0)
    return pickle.dumps({
        'mtype': TRAIN_JOIN,
        'data': {'model_name': model, 'lr': params.lr},
    })


def start_round(W, block, rounds=0, prev_eval=None):
    return pickle.dumps({
        'mtype': TRAIN_START,
        'data': {'W': W, 'block': block, 'rounds': rounds, 'prev_eval': prev_eval},
    })


def stop_train(server):
    battery = np.sum(server.battery_usage[-server.status.active])
    data = {
        'performance': server.performance[-1],
        'battery_usage': battery,
        'iteration_cost': np.mean(server.iteration_cost)
    }
    log('success', data)
    return pickle.dumps({
        'mtype': TRAIN_STOP,
        'data': data,
    })


def disconnect():
    return pickle.dumps({
        'mtype': DISCONNECT,
        'data': {},
    })
