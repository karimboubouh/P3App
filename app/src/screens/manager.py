from kivy.uix.screenmanager import ScreenManager

from src.client import Client


class ScreenManagement(ScreenManager):
    def __init__(self, **kwargs):
        super(ScreenManagement, self).__init__(**kwargs)
        self.client = Client(self)
