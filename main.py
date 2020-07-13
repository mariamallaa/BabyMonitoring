from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout

Builder.load_file('main.kv')

class CameraClick(BoxLayout):
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        
        print("Hello")


class TestCamera(App):

    def build(self):
        return CameraClick()


TestCamera().run()
