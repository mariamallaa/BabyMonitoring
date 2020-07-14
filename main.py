import requests
import flask
import cv2 as cv
import base64
import math

output_file = open("results.txt", "w+")
cap = cv.VideoCapture(
    "C:\\Users\\Maram\\Desktop\\GP2\\labeled dataset\\test\\face and neck.mp4")

requests.post("http://127.0.0.1:5000/setup", data={"Age": 2.5})
frameId = cap.get(1)  # current frame number
frameRate = cap.get(5)  # frame rate
ret, frame = cap.read()
encoded_frame = cv.imencode(".jpg", frame)[1]
requests.post("http://127.0.0.1:5000/newframe",
              data={"Frame": base64.b64encode(encoded_frame)})


#breathing.estimate_breathing_rate(frame, output_file)
while(ret):

    frameId = cap.get(1)  # current frame number
    ret, frame = cap.read()
    if(ret == False):
        break
    if(frameId % math.floor(frameRate) == 0 or frameId % math.floor(frameRate) == math.floor(frameRate/2)):
        encoded_frame = cv.imencode(".jpg", frame)[1]
        requests.post("http://127.0.0.1:5000/newframe",
                      data={"Frame": base64.b64encode(encoded_frame)})
        response = requests.get("http://127.0.0.1:5000/get-breathing-rate")
        print(response.text)

        cv.imshow("sparse optical flow", frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

# print(breathing.prev_rates)
cap.release()
cv.destroyAllWindows()
