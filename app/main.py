from kivy.core.window import Window
from kivy.lang.builder import Builder
from kivymd.app import MDApp

from src.screens import *

Window.size = (336, 600)


class HgOApp(MDApp):

    def build(self):
        return Builder.load_file('src/template.kv')


HgOApp().run()
