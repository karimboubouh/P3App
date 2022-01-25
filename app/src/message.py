import pickle

DISCONNECT = 0
TRAIN_JOIN = 1
TRAIN_START = 2
TRAIN_INFO = 3
TRAIN_STOP = 5


def train_info(grads, gtime, battery):
    return pickle.dumps({
        'mtype': TRAIN_INFO,
        'data': {'grads': grads, 'time': gtime, 'battery': battery}
    })


def disconnect():
    return pickle.dumps({
        'mtype': DISCONNECT,
        'data': {},
    })
