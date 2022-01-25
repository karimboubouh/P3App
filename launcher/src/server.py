import socket
import time

import numpy as np

from conf import SOCK_TIMEOUT, TCP_SOCKET_SERVER_LISTEN, MIN_ACTIVE_WORKERS, SERVER_PORT, SERVER_HOST
from . import aggregators as GARs
from . import message
from .device_connection import DeviceConnection
from .models import LogisticRegression
from .utils import log, create_tcp_socket, input_size, load_params, Map, chunks, load_test_dataset, save


class Server:

    def __init__(self, name="SFServer", model_name="LR", dataset="mnist", params=None, host=SERVER_HOST,
                 port=SERVER_PORT):
        self.name = name
        self.host = host
        self.port = port
        self.sock = None
        self.terminate = False
        self.status = Map({'active': 0, 'train': False, 'aggregate': False, 'started': False})
        self.model_name = model_name
        self.model = None
        self.X = None
        self.y = None
        self.workers = []
        self.tmp_workers = []
        self.blocks = None
        self.current_block = None
        self.grads = []
        self.performance = []
        self.battery_usage = []
        self.iteration_cost = []
        self.byz_indices = []
        self.params = load_params(params)
        self.input_size = input_size(model_name, dataset)
        self.init()

    def init(self):
        self._init_server()
        self._init_model()
        self._init_params()

    def start(self):
        log('info', f"{self.name}: Waiting for incoming connections ...")
        while not self.terminate:
            try:
                conn, address = self.sock.accept()
                if not self.terminate:
                    device_thread = DeviceConnection(self, conn, address)
                    device_thread.start()
                    # Send model and configuration
                    device_thread.send(message.join_train(self.model_name, self.params))
                    if self.status.started:
                        self.tmp_workers.append(device_thread)
                    else:
                        self.workers.append(device_thread)
                    # Start training if enough workers
                    self.update_status()
                    log('info', f"New device <{address}> connected [Total workers: {len(self.workers)}]")
                    self.start_train()
            except socket.timeout as e:
                pass
            except Exception as e:
                log('error', f"{self.name}: Exception1\n{e}")
            self.update_status()

        log('info', f"{self.name}: Terminating connections ...")
        for w in self.workers:
            w.stop()
        time.sleep(1)
        for w in self.workers:
            w.join()
        self.sock.close()
        log('info', f"{self.name}: Stopped.")

    def broadcast(self, msg, only=None):
        try:
            if only:
                for worker in only:
                    worker.send(msg)
            else:
                for worker in self.workers:
                    worker.send(msg)
        except Exception as e:
            log('error', f"{self.name}: Exception2\n{e}")

    def aggregate(self):
        if self.status.aggregate:
            selected_grad = GARs.aggregate(self.grads, self.params.gar)
            # update model
            self.update_model(selected_grad, block=self.current_block)
            # end training round
            prev_eval = self.end_round()
            self.workers.extend(self.tmp_workers)
            self.tmp_workers = []
            # start a new training round
            self.init_new_round(prev_eval)

    def start_train(self):
        if not self.status.started and self.status.train:
            log('info', f"start_train(self) YES")
            # training started
            self.status.started = True
            # choose current block of indices to update
            self.current_block = self.get_block()
            # broadcast round config
            time.sleep(.2)
            self.broadcast(message.start_round(self.model.W, self.current_block))
        else:
            log('info', f"start_train(self) NO")

    def init_new_round(self, prev_eval=None):
        # verify stop condition
        if self.params.rounds > 0:
            # choose current block of indices to update
            self.current_block = self.get_block()
            # broadcast round config
            self.broadcast(message.start_round(self.model.W, self.current_block, self.params.rounds, prev_eval))
        else:
            # broadcast stop training
            log('success', "Training finished")
            self.broadcast(message.stop_train(self))
            save(f"DATA", self.performance)

    def end_round(self):
        self.params.rounds -= 1
        self.grads = []
        self.update_status()
        cost, acc = self.model.evaluate(self.X, self.y)
        # self.performance.append((cost, acc))
        self.performance.append(acc)
        log('result', f" {self.params.rounds} rounds to finish | Cost: {round(cost, 8)} | Accuracy: {round(acc, 8)}")

        return cost, acc

    def evaluate(self):
        return self.model.evaluate(self.X, self.y)

    def summary(self, X, y):
        print(f"------------------- Training for {self.params.rounds} rounds. using <{self.params.gar}>")
        print(f">> Train: loss: {self.history[-1][0]:.4f}, Accuracy: {(self.history[-1][1] * 100):.2f}%.")
        return self

    def active_workers(self):
        return [worker for worker in self.workers if worker.terminate is False]

    def get_block(self):
        if self.params.block_size:
            index = np.random.choice(len(self.params.block_size), replace=False)
            return self.blocks[index]
        else:
            return None

    def update_model(self, grad, block):
        grad = grad.reshape(-1, 1)
        if block is None:
            self.model.W = self.model.W - self.params.lr * grad
        else:
            self.model.W[block] = self.model.W[block] - self.params.lr * grad

    def update_status(self):
        self.status.active = len(self.active_workers())
        self.status.train = True if self.status.active >= MIN_ACTIVE_WORKERS else False
        self.status.aggregate = self.status.train and len(self.grads) >= self.params.frac * self.status.active

    # -------------------------------------------------------------------------
    def _init_server(self):
        log('info', f"Starting server on ({self.host}:{self.port})")
        self.sock = create_tcp_socket()
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(SOCK_TIMEOUT)
        self.sock.listen(TCP_SOCKET_SERVER_LISTEN)

    def _init_model(self):
        if self.model_name == 'LR':
            self.X, self.y = load_test_dataset("mnist")
            # print(f"self.X == {self.X.shape}, self.y == {self.y.shape}")
            self.model = LogisticRegression(self.input_size, lr=self.params.lr)
        else:
            log('error', f"Model {self.model_name} not supported.")
            exit(0)

    def _init_params(self):

        self.blocks = chunks(list(range(self.input_size)), self.params.block_size)

    # Special methods
    def __repr__(self):
        return f"Server()"

    def __str__(self):
        return f"Server()"
