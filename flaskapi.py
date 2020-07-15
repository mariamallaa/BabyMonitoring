import flask
from flask import request, jsonify
from breathing_frame import *
import cv2 as cv
import base64
from Danger_zone.Danger_Zone_integrated import *
from Danger_zone.Face_covered import *
from Danger_zone.bounding_box import *

import base64
import webbrowser

import PIL.Image
import numpy as np


app = flask.Flask(__name__)
app.config["DEBUG"] = True

cam_width = 0
cam_height = 0
html_opened = False

feature_params = dict(maxCorners=100, qualityLevel=0.05,
                      minDistance=30, blockSize=3)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


breathing_module = breathing_rate(feature_params, lk_params)
# breathing_rate(feature_params, lk_params, 2.5)
Danger_zone_module=DangerZone()
Face_covered_module=CoverFace()


@app.route('/camSize', methods = ['GET', 'POST'])
def cam_size():
    global cam_width
    global cam_height
    cam_width = int(float(flask.request.args["width"]))
    cam_height = int(float(flask.request.args["height"]))
    print('Width',cam_width,'& Height',cam_height,'Received Successfully.')
    return "OK"

@app.route('/', methods = ['POST'])
def upload_file():
    global cam_width
    global cam_height
    global html_opened
    file_to_upload = flask.request.files['media'].read()
    #print('File Uploaded Successfully.')
    im_base64 = base64.b64encode(file_to_upload)

    image = PIL.Image.frombytes(mode="RGBA", size=(cam_width, cam_height),data=file_to_upload)

    imwhole = np.array(image)

    frame = cv.cvtColor(imwhole, cv.COLOR_RGB2BGR)
    danger_zone_coor=Danger_zone_module.getDangerZone()
    point1=(cam_width-danger_zone_coor[0],cam_height-danger_zone_coor[1])
    point2=(cam_width-danger_zone_coor[2],cam_height-danger_zone_coor[3])

    
    


    Danger_zone_module.Danger_zone(frame,False)
    if(Face_covered_module.return_first_time()):
 
        Face_covered_module.get_first_frame(frame)
    else:

        Face_covered_module.detect_covered(frame)
    # cFv.imwrite("frame.jpg", frame)
    #breathing_module.estimate_breathing_rate(frame)
    
    if(Face_covered_module.is_face_same()):
        breathing_module.estimate_breathing_rate(frame)
    else:
        print("stop breathing rate module")
    
    #Face_covered_module.set_oldface()
    point1=(0,0)
    point2=(cam_width,cam_height)
    cv.rectangle(frame, point1, point2, (0, 255, 0), 2)
    
    cv.imshow("image", frame)

    cv.waitKey(0)

    return "SUCCESS"

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


@app.route("/setup", methods=["POST"])
def initiate():
    data = request.form
    print("dataaa")
    print(data)
    age = float(data["Age"])
    dangerzone=[int(data["x0"]),int(data["y0"]),int(data["x1"]),int(data["y1"])]
    print("dangerzone")
    print(dangerzone)
    
    breathing_module.set_age(age)
    print(breathing_module.age)
    #print(dangerzone)
    Danger_zone_module.setDangerZone(dangerzone)

    return "ok"


@app.route("/newframe", methods=["POST"])
def new_frame():
    data = request.form
    string = data['Frame']
    img = base64.b64decode(string)
    img_as_np = np.frombuffer(img, dtype=np.uint8)
    frame = cv.imdecode(img_as_np, flags=1)
    # print(frame.shape)

    Danger_zone_module.Danger_zone(frame,False)
    if(Face_covered_module.return_first_time()):
        #print("hereeeeeeeeeeeeeeeeee")
        Face_covered_module.get_first_frame(frame)
    else:
        Face_covered_module.detect_covered(frame)
    # cv.imwrite("frame.jpg", frame)
    breathing_module.estimate_breathing_rate(frame)

    return "received"

'''
@app.route('/get-breathing-rate', methods=['GET'])
def get_rate():
    rate, state = breathing_module.get_breathing_rate()

    return {"rate": rate, "state": state}
'''

@app.route('/get-stats', methods=['GET'])
def get_state():
    rate, state = breathing_module.get_breathing_rate()
    indanger=Danger_zone_module.Is_Danger()
    iscovered=False
    if(Face_covered_module!=None):
        iscovered=Face_covered_module.Is_Covered()
        #print("is coveredddddd")
        #print(iscovered)
    return {"rate": rate, "state": state, "indanger":indanger,"iscovered":iscovered}


app.run()
