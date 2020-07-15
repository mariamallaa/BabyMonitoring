import flask
from flask import request, jsonify
from breathing_frame import *
import cv2 as cv
import base64
from Danger_zone.Danger_Zone_integrated import *
from Danger_zone.Face_covered import *
from Danger_zone.bounding_box import *

app = flask.Flask(__name__)
app.config["DEBUG"] = True

feature_params = dict(maxCorners=100, qualityLevel=0.05,
                      minDistance=30, blockSize=3)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


breathing_module = breathing_rate(feature_params, lk_params)
# breathing_rate(feature_params, lk_params, 2.5)
Danger_zone_module=DangerZone()
Face_covered_module=CoverFace()

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


@app.route("/setup", methods=["POST"])
def initiate():
    data = request.form
    age = float(data["Age"])
    dangerzone=[int(data["x0"]),int(data["y0"]),int(data["x1"]),int(data["y1"])]
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
        print("hereeeeeeeeeeeeeeeeee")
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
    return {"rate": rate, "state": state, "indanger":indanger,"iscovered":iscovered}


app.run()
