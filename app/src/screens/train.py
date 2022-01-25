from threading import Thread

from kivy.uix.screenmanager import Screen


class TrainScreen(Screen):
    def __init__(self, **kwargs):
        self.logs = {'log': '', 'cpu': '', 'energy': '', 'net': '', 'acc': ''}
        super(TrainScreen, self).__init__(**kwargs)

    def init_train(self, *args):
        self.update_log(log="Waiting for training to start...")
        Thread(target=self.train).start()

    def train(self):
        pass

    def log_info(self):
        pass

    def disconnect(self):
        self.manager.client.disconnect()
        self.manager.current = 'welcome'

    def update_log(self, log=None, cpu=None, energy=None, acc=None):
        self.logs = {'log': '', 'cpu': '', 'energy': '', 'net': '', 'acc': ''}
        if log: self.logs['log'] = log
        if cpu is not None: self.logs['cpu'] = cpu
        if energy is not None: self.logs['energy'] = energy
        if acc: self.logs['acc'] = acc
        log = ''
        if self.logs['log'] != '': log += f"{self.logs['log']}\n"
        if self.logs['cpu'] != '': log += f"Iteration cost: {self.logs['cpu']}\n"
        if self.logs['energy'] != '': log += f"Battery usage: {self.logs['energy']}\n"
        if self.logs['acc'] != '': log += f"Current train accuracy: {round(self.logs['acc'][1], 6)}\n"
        self.ids.train_log.text = log
