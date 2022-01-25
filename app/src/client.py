import plyer

from conf import DEFAULT_PROFILE, SOCK_TIMEOUT, DEFAULT_BATTERY_CAPACITY
from . import message
from .client_thread import ClientThread
from .utils import Map, create_tcp_socket, mah


class Client:

    def __init__(self, manager, profile=DEFAULT_PROFILE, train=None, test=None):
        self.id = plyer.uniqueid.id  # .decode("utf-8")
        self.manager = manager
        self.profile = profile
        self.sock = None
        self.listener = None
        self.byzantine = None
        self.model = None
        self.grads = None
        self.iteration_cost = []
        self.train = train
        self.test = test
        self.battery_capacity = DEFAULT_BATTERY_CAPACITY
        self.battery_start = plyer.battery.status['percentage']
        # default params
        self.params = Map({
            'lr': 1,
            'block': 5
        })

    def connect(self, host, port):
        try:
            self.sock = create_tcp_socket()
            self.sock.settimeout(SOCK_TIMEOUT)
            self.sock.connect((host, port))
            self.listener = ClientThread(self)
            self.listener.start()
            return True
        except Exception:
            return False

    def disconnect(self):
        self.listener.send(message.disconnect())
        self.listener.stop()
        self.sock.close()

    def local_train(self, data):
        self.model.W = data['W']
        self.grads, gtime = self.model.one_epoch(self.train.data, self.train.targets, data['block'])
        battery_usage = mah(self.battery_start, self.battery_capacity)
        self.listener.send(message.train_info(self.grads, gtime, battery_usage))
        self.iteration_cost.append(gtime)
        self.log(
            log=f"Training... | Remaining rounds: {data['rounds']}",
            cpu=round(gtime, 8),
            energy=round(battery_usage, 8),
            acc=data['prev_eval']
        )

    def attack(self):
        return self.byzantine.attack()

    def log(self, log=None, cpu=None, energy=None, acc=None):
        self.manager.get_screen("train").update_log(log, cpu, energy, acc)

    # Special methods
    def __repr__(self):
        return f"Worker ({self.id})"

    def __str__(self):
        return f"Client ({self.id})"
