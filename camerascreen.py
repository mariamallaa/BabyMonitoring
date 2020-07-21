import kivy.app
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen


class CameraScreen(Screen):
     def __init__(self, **kwargs):
          super(CameraScreen, self).__init__(**kwargs)
          self.num_images = 0
          self.danger_zone_coordinates=[]
          self.label = Label(text=" ",color=(1,0,0,1), font_size=(20),size_hint=(0.2,0.1), pos_hint={"center_x":0.1, "center_y":0.1})

     def on_touch_down(self, touch):
          self.danger_zone_coordinates=[]
          print(touch)
          self.danger_zone_coordinates.append([int(touch.pos[0]),int(touch.pos[1])])
          return super(CameraScreen, self).on_touch_down(touch)

 
     def on_touch_up(self, touch):
          self.danger_zone_coordinates.append([int(touch.pos[0]),int(touch.pos[1])])
          print("RELEASED!",self.danger_zone_coordinates)
          self.label.text=str(touch.pos[0])
          self.add_widget(self.label)
          return super(CameraScreen, self).on_touch_up(touch)
        
    