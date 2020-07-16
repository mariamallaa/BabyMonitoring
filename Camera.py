import kivy.app
import requests
import kivy.clock
import threading
from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout

class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
        self.num_images = 0
        self.danger_zone_coordinates=[]
   
        
    def cam_size(self):
       
        camera = self.root.ids['camera']
        cam_width_height = {'width': camera.resolution[0], 'height':camera.resolution[1]}
        setup_data={ "x0":self.danger_zone_coordinates[0][0], "y0":self.danger_zone_coordinates[0][1], "x1":self.danger_zone_coordinates[1][0], "y1":self.danger_zone_coordinates[1][1]}
        ip_addr = self.root.ids['ip_address'].text
        #port_number = self.root.ids['port_number'].text
        url = 'http://' + ip_addr  
        try:
            self.root.ids['cam_size'].text = "Trying to Establish a Connection..."
            requests.post(url+  '/camSize', params=cam_width_height)
            self.root.ids['cam_size'].text = "Done."
            self.root.current = "capture"
            requests.post(url+'/setup-bb', data=setup_data)
        except requests.exceptions.ConnectionError:
            self.root.ids['cam_size'].text = "Connection Error! Make Sure Server is Active."

    def capture(self):
        kivy.clock.Clock.schedule_interval(self.upload_images, 0.5)

    def upload_images(self,*args):

        self.num_images = self.num_images + 1
        print("Uploading image", self.num_images)
        camera = self.root.ids['camera']
        print("Image Size ", camera.resolution[0], camera.resolution[1])
        print("Image corner ", camera.x, camera.y)
        pixels_data = camera.texture.get_region(x=0, y=0,width=camera.resolution[0], height=camera.resolution[1]).pixels
        ip_addr = self.root.ids['ip_address'].text
        #port_number = self.root.ids['port_number'].text
        url = 'http://' + ip_addr  + '/'
        response = requests.get(url+"get-stats")
        print(response.text)
        files = {'media': pixels_data}
        t = threading.Thread(target=self.send_files_server, args=(files, url))
        t.start()

    def send_files_server(self, files, url):
        try:
            requests.post(url, files=files)
            response = requests.get(url+"get-stats")
            print(response.text)
        except requests.exceptions.ConnectionError:
            self.root.ids['capture'].text = "Connection Error! Make Sure Server is Active."

    
    def on_touch_down(self, touch):
        self.danger_zone_coordinates=[]
        print(touch)
        self.danger_zone_coordinates.append([int(touch.pos[0]),int(touch.pos[1])])

 
    def on_touch_up(self, touch):
        self.danger_zone_coordinates.append([int(touch.pos[0]),int(touch.pos[1])])
        print("RELEASED!",self.danger_zone_coordinates)

    def getstat(self):
        while True:
            ip_addr = self.root.ids['ip_address'].text
            #port_number = self.root.ids['port_number'].text
            url = 'http://' + ip_addr  + '/'

            try:
                self.root.ids['capture'].text = "Trying to Establish a Connection..."
                response = requests.get(url+"get-stats")
                print(response.text)
            except requests.exceptions.ConnectionError:
                self.root.ids['capture'].text = "Connection Error! Make Sure Server is Active."
        

    
