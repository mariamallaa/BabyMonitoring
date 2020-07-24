from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
    

import kivy.app
import requests
import kivy.clock
import threading
import urllib3
from kivy.uix.screenmanager import Screen
from kivy.app import App



class PlayingScreen(Screen):
    def __init__(self, **kwargs):
        super(PlayingScreen, self).__init__(**kwargs)
        self.num_images = 0
        self.danger_zone_coordinates = []

    def cam_sizeplay(self):
        print("cam size")
        camera = self.ids['cameraplay']
        cam_width_height = {'width': camera.resolution[0], 'height': camera.resolution[1]}
        setup_data = {"x0": self.danger_zone_coordinates[0][0], "y0": self.danger_zone_coordinates[0]
                      [1], "x1": self.danger_zone_coordinates[1][0], "y1": self.danger_zone_coordinates[1][1]}
        ip_addr = self.ids['ip_addressplay'].text
        #port_number = self.ids['port_number'].text
        url = 'http://' + ip_addr
        try:
            self.ids['cam_sizeplay'].text = "Trying to Establish a Connection..."
            requests.post(url + '/camSize', params=cam_width_height)
            self.ids['cam_sizeplay'].text = "Done."
            self.current = "capture"
            print("setting up BB")
            requests.post(url+'/setup-bb', data=setup_data)
        except requests.exceptions.ConnectionError:
            self.ids['cam_sizeplay'].text = "Connection Error! Make Sure Server is Active."

    def captureplay(self):
        kivy.clock.Clock.schedule_interval(self.upload_imagesplay, 0.5)

    def upload_imagesplay(self, *args):

        self.num_images = self.num_images + 1
        print("Uploading image", self.num_images)
        camera = self.ids['cameraplay']
        print("Image Size ", camera.resolution[0], camera.resolution[1])
        print("Image corner ", camera.x, camera.y)
        pixels_data = camera.texture.get_region(
            x=0, y=0, width=camera.resolution[0], height=camera.resolution[1]).pixels
        ip_addr = self.ids['ip_addressplay'].text
        #port_number = self.ids['port_number'].text
        url = 'http://' + ip_addr + '/playing'
        response = requests.get(url+"get-stats")
        print(response.text)
        files = {'media': pixels_data}
        url = 'http://' + ip_addr + '/'
        t = threading.Thread(target=self.send_files_serverplay, args=(files, url))
        t.start()

    def send_files_serverplay(self, files, url):
        try:
            requests.post(url, files=files)
            response = requests.get(url+"get-stats")
            print(response.text)
        except requests.exceptions.ConnectionError:
            self.ids['captureplay'].text = "Connection Error! Make Sure Server is Active."

    def on_touch_down(self, touch):
        print(touch)
        touch.apply_transform_2d(self.to_local)
        print(touch)
        # self.danger_zone_coordinates=[]
        print("touch")
        print(touch)
        self.danger_zone_coordinates.append([touch.spos[0], touch.spos[1]])
        return super(PlayingScreen, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        print(touch)
        touch.apply_transform_2d(self.to_local)
        print(touch)
        self.danger_zone_coordinates.append([touch.spos[0], touch.spos[1]])
        print("RELEASED!", self.danger_zone_coordinates)
        self.cam_sizeplay()
        # self.capture()
        return super(PlayingScreen, self).on_touch_up(touch)
    def on_enter(self):
        self.ids['cameraplay'].play = True