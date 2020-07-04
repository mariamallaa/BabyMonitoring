import numpy as np
import argparse
import cv2
from commonfunctions import *
prototxt='C:/Users/Mariam Alaa/Documents/GitHub/BabyMonitoringIntegration/weights/deploy.prototxt.txt'
caffemodel='C:/Users/Mariam Alaa/Documents/GitHub/BabyMonitoringIntegration/weights/res10_300x300_ssd_iter_140000.caffemodel'
def get_face_BB(image):
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    (h, w) = image.shape[:2]
    #print("probbbbbbbbb")
    #show_images([image])

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	    (300, 300), (104.0, 177.0, 123.0))
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    dimensions=[]
    #print(detections)
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        #print(confidence)
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
    
            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            dimensions.append([startX,startY,endX,endY])
    #print(dimensions)
    return dimensions


#image = cv2.imread("Capture2.PNG")
#dim=get_face_BB(image)
#print(dim)
#cv2.imshow("Output", image)
#cv2.waitKey(0)