
import kivy.app
import requests
import kivy.clock
import threading
import urllib3
from kivy.uix.screenmanager import Screen
from kivy.app import App
from kivy.properties import StringProperty
import json
from kivy.core.audio import SoundLoader

sound = SoundLoader.load('alarm.mp3')
class CameraScreen(Screen):

    breathingrate = StringProperty()
    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
        self.num_images = 0
        self.danger_zone_coordinates=[]
   
        
    def cam_size(self):
        print("cam size")
        camera = self.ids['camera']
        cam_width_height = {'width': camera.resolution[0], 'height':camera.resolution[1]}
        setup_data={ "x0":self.danger_zone_coordinates[0][0], "y0":self.danger_zone_coordinates[0][1], "x1":self.danger_zone_coordinates[1][0], "y1":self.danger_zone_coordinates[1][1]}
        ip_addr ='192.168.1.102:8080'
        #port_number = self.ids['port_number'].text
        url = 'http://' + ip_addr  
        try:
            text = "Trying to Establish a Connection..."
            requests.post(url+  '/camSize', params=cam_width_height)
            #self.ids['camsize'].text = "Done."
            self.current = "capture"
            print("setting up BB")
            requests.post(url+'/setup-bb', data=setup_data)
        except requests.exceptions.ConnectionError:
            text = "Connection Error! Make Sure Server is Active."


    def capture(self):
        kivy.clock.Clock.schedule_interval(self.upload_images, 0.5)

    def upload_images(self, *args):

        self.num_images = self.num_images + 1
        print("Uploading image", self.num_images)
        camera = self.ids['camera']
        print("Image Size ", camera.resolution[0], camera.resolution[1])
        print("Image corner ", camera.x, camera.y)
        pixels_data = camera.texture.get_region(
            x=0, y=0, width=camera.resolution[0], height=camera.resolution[1]).pixels
        ip_addr ='192.168.1.102:8080'
        #port_number = self.ids['port_number'].text
        url = 'http://' + '192.168.1.102:8080'+ '/'
        response = requests.get(url+"get-stats")
        print(response.text)
        parsed_data = json.loads(response.text)
        breathingrate=parsed_data['rate']
        if(breathingrate!=None):
            self.ids['stats'].text="Breathing Rate:  "+str(int(breathingrate))
        else:
            self.ids['stats'].text="Breathing Rate:  "

        if sound:
            print("indangeeeeeeeeeeeer")
            print(parsed_data['indanger'])
            if(parsed_data['indanger']==True or parsed_data['iscovered']==True or  parsed_data['state']=='Danger'):
                sound.play()
        files = {'media': pixels_data}
        t = threading.Thread(target=self.send_files_server, args=(files, url))
        t.start()

    def send_files_server(self, files, url):
        try:
            requests.post(url, files=files)
            response = requests.get(url+"get-stats")
            print(response.text)
        except requests.exceptions.ConnectionError:
            text = "Connection Error! Make Sure Server is Active."

    def on_touch_down(self, touch):
        print(touch)
        touch.apply_transform_2d(self.to_local)
        print(touch)
        # self.danger_zone_coordinates=[]
        print("touch")
        print(touch)
        self.danger_zone_coordinates.append([touch.spos[0], touch.spos[1]])
        return super(CameraScreen, self).on_touch_down(touch)
 
    def on_touch_up(self, touch):
        print(touch)
        touch.apply_transform_2d(self.to_local)
        print(touch)
        self.danger_zone_coordinates.append([touch.spos[0], touch.spos[1]])
        print("RELEASED!", self.danger_zone_coordinates)
        self.cam_size()
        #self.capture()
        return super(CameraScreen, self).on_touch_up(touch)

    def getstat(self):
        while True:
            ip_addr ='192.168.1.102:8080'
            #port_number = self.ids['port_number'].text
            url = 'http://' + ip_addr + '/'

            try:
                text = "Trying to Establish a Connection..."
                print("updating")
                response = requests.get(url+"get-stats")
                print(response.text)
                parsed_data = json.loads(response.text)
                breathingrate=parsed_data['rate']
                #print(parsed_data['image'])
                self.clear_widgets(self.ids["stats"])
                stats = ThemedButton(text=breathingrate)
                self.add_widget(stats)
                self.ids['stats'].text=response.text

            except requests.exceptions.ConnectionError:
                text = "Connection Error! Make Sure Server is Active."




             