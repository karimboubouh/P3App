import os
from threading import Thread
from time import sleep

from kivy.clock import Clock
from kivy.uix.screenmanager import Screen
from kivymd.toast import toast
from kivymd.uix.dialog import MDDialog
from kivymd.uix.filemanager import MDFileManager
from kivy.utils import platform

from conf import DEFAULT_NB_SAMPLES, DEFAULT_PROFILE, SERVER_HOST, SERVER_PORT, DEFAULT_BATTERY_CAPACITY
from src.utils import sample_data, Map, mnist


class ConfScreen(Screen):

    def __init__(self, **kwargs):
        self.dialog = None
        self.nb_samples = DEFAULT_NB_SAMPLES
        self.battery_capacity = DEFAULT_BATTERY_CAPACITY
        self.profile = DEFAULT_PROFILE
        self.host = SERVER_HOST
        self.port = SERVER_PORT
        self.dataset_path = ""
        super(ConfScreen, self).__init__(**kwargs)
        Clock.schedule_once(self.init, 1)
        self.file_manager = MDFileManager(
            select_path=self.select_path,
            # preview=True,
        )

    def init(self, *args):
        self.ids.server_host.text = self.host
        self.ids.server_port.text = str(self.port)
        self.ids.samples_input.text = str(self.nb_samples)
        self.ids.battery_capacity.text = str(self.battery_capacity)

    def file_manager_open(self):
        path_root = '/storage/emulated/0/' if platform == 'android' else '/'
        self.file_manager.show(path_root)

    def select_path(self, path):
        self.file_manager.close()
        self.dataset_path = path
        self.ids.ds_label.text = f"Selected dataset: {os.path.basename(path)}"
        toast(path)

    def configure(self):
        self.dialog = MDDialog(title="Initializing ...", auto_dismiss=False)
        self.read_configuration()
        Thread(target=self.init_train).start()

    def read_configuration(self):
        try:
            self.nb_samples = int(self.ids['samples_input'].text)
            self.host = self.ids['server_host'].text
            self.port = int(self.ids['server_port'].text)
            self.nb_samples = int(self.ids['samples_input'].text)
            self.battery_capacity = int(self.ids['battery_capacity'].text)
            if self.ids['mod_cap'].state == "down":
                self.profile = "mod"
            elif self.ids['pow_cap'].state == "down":
                self.profile = "pow"
            else:
                self.profile = "low"
        except ValueError:
            self.nb_samples = DEFAULT_NB_SAMPLES
            self.profile = DEFAULT_PROFILE

    def init_train(self):
        self.manager.client.battery_capacity = self.battery_capacity
        self.dialog.open()
        if not self.load_data():
            sleep(.5)
            self.dialog.dismiss()
            self.manager.current = 'conf'
            return
        if not self.connect_server():
            sleep(.5)
            self.dialog.dismiss()
            self.manager.current = 'welcome'
            return
        self.dialog.dismiss()
        self.manager.current = 'train'

    def load_data(self):
        try:
            self.dialog.title = "Loading data"
            ds = os.path.basename(self.dataset_path)
            self.dialog.text = f"Load {self.nb_samples} data points from dataset: {ds} ..."
            X_train, Y_train = mnist(self.dataset_path)
            full_train = Map({'data': X_train, 'targets': Y_train})
            self.manager.client.train = sample_data(full_train, self.nb_samples)
            self.dialog.text = f"Data loaded successfully"
            return True
        except TypeError:
            self.dialog.text = f"No dataset have been selected"
            return False

    def connect_server(self):
        try:
            self.dialog.title = "Connection to server"
            self.dialog.text = f"Connecting ..."
            if self.manager.client.connect(self.host, self.port):
                self.dialog.text = f"Connected successfully"
                return True
            else:
                self.dialog.text = f"Could not connect to server."
                return False
        except Exception as e:
            self.dialog.text = f"Error while connecting to server: {str(e)}"
            return False
