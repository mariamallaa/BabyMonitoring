import cv2
import time
import numpy as np
import tensorflow as tf

from model.model import create_model
from config import cfg
from draw_boxes import draw_outputs
from skimage.transform import resize

weights_dir="./model3-0.41.h5"
SCORE_THRESHOLD = 0.5
MAX_OUTPUT_SIZE = 49

def detectface(frame):
    
    
    image = tf.keras.preprocessing.image.load_img("fullfull.PNG",target_size=(224 , 224 ))
    
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image - tf.keras.applications.mobilenet_v2.preprocess_input(image)
    original_image=image.copy()
    t1 = time.time()
    pred = np.squeeze(model.predict(image))
    t2 = time.time()
    processing_time = t2 - t1
    height, width, y_f, x_f, score = [a.flatten() for a in np.split(pred, pred.shape[-1], axis=-1)]

    coords = np.arange(pred.shape[0] * pred.shape[1])
    y = (y_f + coords // pred.shape[0]) / pred.shape[0]
    x = (x_f + coords % pred.shape[1]) / pred.shape[1]

    boxes = np.stack([y, x, height, width, score], axis=-1)
    boxes = boxes[np.where(boxes[..., -1] >= SCORE_THRESHOLD)]
    original_image = draw_outputs(original_image, boxes)
    original_image = cv2.putText(original_image, "Time: {:.2f}".format(processing_time), (0, 30),
                      cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    
    return original_image

model = create_model()
model.load_weights(weights_dir)
cam = cv2.VideoCapture("WIN_20200704_15_14_19_Pro.mp4") 
while(True): 
    # reading from frame 
    ret,frame = cam.read()
    
    frame = resize(frame, (224, 224))
    if not ret: 
        break
    newframe=detectface(frame)
    cv2.imshow('Main',newframe)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    