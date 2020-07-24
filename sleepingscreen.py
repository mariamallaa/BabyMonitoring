from kivy.uix.screenmanager import Screen
from kivy.uix.image import Image
from kivy.config import Config
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import BooleanProperty
import requests
import urllib3


class SleepingScreen(Screen):
    age_right = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(SleepingScreen, self).__init__(**kwargs)
        self.label = Label(text=" ", color=(1, 0, 0, 1), font_size=(
            20), size_hint=(0.2, 0.1), pos_hint={"center_x": 0.5, "center_y": 0.9})
        self.add_widget(self.label)

    def on_age_right(self, *args):
        print("right age!")

    def check_user_input(self, age):

        if age == '':
            self.label.text = "Please re-enter !"
            return self.__init__()

        elif age.isdigit() == False:
            self.label.text = "Please re-enter !"
            return self.__init__()

        else:
            self.age_right = True
            url = "http://"+"192.168.1.102:8080"+'/setup-age'
            requests.post(url, data={'Age': age})
