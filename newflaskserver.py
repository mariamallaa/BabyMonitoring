import flask
import base64
import webbrowser
import cv2
import PIL.Image

app = flask.Flask(import_name="FlaskUpload")
cam_width = 0
cam_height = 0
html_opened = False
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
    print('File Uploaded Successfully.')
    im_base64 = base64.b64encode(file_to_upload)

    image = PIL.Image.frombytes(mode="RGBA", size=(cam_width, cam_height),data=file_to_upload)
    import numpy as np
    imwhole = np.array(image)
    print(imwhole)

    imcv = cv2.cvtColor(imwhole, cv2.COLOR_RGB2BGR)
    cv2.imshow("image", imcv)

    cv2.waitKey(0)
    html_code = '<html><head><meta http-equiv="refresh" content="0.5"><title>Displaying Uploaded Image</title></head><body><h1>UploadedImage to the Flask Server</h1><img src="data:;base64,'+im_base64.decode('utf8')+'" alt="Uploaded Image at the Flask Server"/></body></html>'
    html_url = "test.html"
    f = open(html_url,'w')
    f.write(html_code)
    f.close()
    if html_opened == False:
        webbrowser.open(html_url)
        html_opened = True
    return "SUCCESS"
app.run()
