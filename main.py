from kivy.app import App
from kivy import utils
import certifi
import os

class MainApp(App):
        #login_primary_color = utils.get_color_from_hex("#ABCDEF")#(1, 0, 0, 1)
        #login_secondary_color = utils.get_color_from_hex("#060809")#(1, 1, 0, 1)
        #login_tertiary_color = utils.get_color_from_hex("#434343")#(0,0, 1, 1)
    def build(self):
        self.title ="SafeAndSound"
      

if __name__ == "__main__":
    MainApp().run()
