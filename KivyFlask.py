from kivy.app import App
import requests
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout

Builder.load_file('KivyFlask.kv')

class cameraTest(BoxLayout):
    def cam_size(self): 
        camera=self.root.ids['camera']  
        cam_width_height = {'width': camera.resolution[0], 'height': camera.resolution[1]}
        ip_addr = self.root.ids['ip_address'].text  
        url = 'http://'+ip_addr+':6666/camSize'
        try:         
            self.root.ids['cam_size'].text = "Trying to Establish a Connection..."   
            requests.post(url, params=cam_width_height)        
            self.root.ids['cam_size'].text = "Done."    
            self.root.remove_widget(self.root.ids['cam_size'])       
        except requests.exceptions.ConnectionError:          
           self.root.ids['cam_size'].text = "Connection Error! Make Sure Server is Active."

    def capture(self):      
        camera = self.root.ids['camera']   
        print(camera.x, camera.y)

        pixels_data = camera.texture.get_region(x=camera.x, y=camera.y, width=camera.resolution[0], height=camera.resolution[1]).pixels
        ip_addr = self.root.ids['ip_address'].text     
        url = 'http://'+ip_addr+':6666/'        
        files = {'media': pixels_data}

        try:            
            self.root.ids['capture'].text = "Trying to Establish a Connection..."      
            requests.post(url, files=files)     
            self.root.ids['capture'].text = "Capture Again!"     
        except requests.exceptions.ConnectionError:   
            self.root.ids['capture'].text = "Connection Error! Make Sure Server is Active."

class PycamApp(App):
    def build(self): 
        return cameraTest() 
    def cam_size(self): 
        camera=self.root.ids['camera']  
        cam_width_height = {'width': camera.resolution[0], 'height': camera.resolution[1]}
        ip_addr = self.root.ids['ip_address'].text  
        url = 'http://'+ip_addr+':6666/camSize'
        try:         
            self.root.ids['cam_size'].text = "Trying to Establish a Connection..."   
            requests.post(url, params=cam_width_height)        
            self.root.ids['cam_size'].text = "Done."    
            self.root.remove_widget(self.root.ids['cam_size'])       
        except requests.exceptions.ConnectionError:          
           self.root.ids['cam_size'].text = "Connection Error! Make Sure Server is Active."

    def capture(self):      
        camera = self.root.ids['camera']   
        print(camera.x, camera.y)

        pixels_data = camera.texture.get_region(x=camera.x, y=camera.y, width=camera.resolution[0], height=camera.resolution[1]).pixels
        ip_addr = self.root.ids['ip_address'].text     
        url = 'http://'+ip_addr+':6666/'        
        files = {'media': pixels_data}

        try:            
            self.root.ids['capture'].text = "Trying to Establish a Connection..."      
            requests.post(url, files=files)     
            self.root.ids['capture'].text = "Capture Again!"     
        except requests.exceptions.ConnectionError:   
            self.root.ids['capture'].text = "Connection Error! Make Sure Server is Active."
        
app=PycamApp() 
app.run()