import cv2
import time
import numpy as np
import tensorflow as tf


from config import cfg
from draw_boxes import draw_outputs
from skimage.transform import resize
from PIL import Image
size = (224, 224)

weights_dir="./model3-0.41.h5"
SCORE_THRESHOLD = 0.5
MAX_OUTPUT_SIZE = 49
model = create_model()
model.load_weights(weights_dir)
import tensorflow as tf
from config import cfg




def create_model(trainable=False):
    base = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE, 3),
                                                          alpha=cfg.NN.ALPHA, weights='imagenet', include_top=False)

    for layer in base.layers:
        layer.trainable = trainable

    out = base.get_layer('block_16_project_BN').output
    # Change 112 to whatever is the size of block_16_project_BN, "112" value is correct for 0.35 ALPHA, 448 is for 1.4
    # Depends on your output complexity you might want to add another Conv2D layers (like one commented out displayed below)
    out = tf.keras.layers.Conv2D(240, padding="same", kernel_size=3, strides=1, activation="relu")(out)
    # out = tf.keras.layers.Conv2D(240, padding="same", kernel_size=3, strides=1, use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)

    out = tf.keras.layers.Conv2D(5, padding='same', kernel_size=1, activation='sigmoid')(out)

    model = tf.keras.Model(inputs=base.input, outputs=out)

    # divide by 2 since d/dweight learning_rate * weight^2 = 2 * learning_rate * weight
    # see https://arxiv.org/pdf/1711.05101.pdf
    regularizer = tf.keras.regularizers.l2(cfg.TRAIN.WEIGHT_DECAY / 2)

    for weight in model.trainable_weights:
        with tf.keras.backend.name_scope('weight_regularizer'):
            model.add_loss(lambda: regularizer(weight))

    return model



def draw_outputs(img, boxes, draw_labels=True):
    dimensions=[]
    for y_c, x_c, h, w, score in boxes:
        x0 = img.shape[1] * (x_c )
        y0 = img.shape[0] * (y_c )
        x1 = x0 + img.shape[1] *w
        y1 = y0 + img.shape[0] *h
        x1y1 = tuple(np.array([x0, y0]).astype(np.int32))
        x2y2 = tuple(np.array([x1, y1]).astype(np.int32))
        print("boxesssssssssssssssssssssssssssss")
        print(x1y1,x2y2)
        dimensions.append([x0,y0,x1,y1])
    return dimensions


def detectface_RM(frame):
    
    
    #image = tf.keras.preprocessing.image.load_img("fullfull.PNG",target_size=(224 , 224 ))
    image2=frame.copy()
    image2 = tf.keras.preprocessing.image.array_to_img(image2)
    image2.thumbnail(size, Image.ANTIALIAS)
    image2 = tf.keras.preprocessing.image.img_to_array(image2)
    
    
    imagenew = np.expand_dims(image2, axis=0)
    imagenew - tf.keras.applications.mobilenet_v2.preprocess_input(imagenew)
    print(imagenew)
    '''
    imagenew = tf.cast(imagenew, tf.float32)
    imagenew = tf.keras.applications.mobilenet_v2.preprocess_input(imagenew)
    '''
    original_image=frame.copy()
    t1 = time.time()
    pred = np.squeeze(model.predict(imagenew))
    t2 = time.time()
    processing_time = t2 - t1
    height, width, y_f, x_f, score = [a.flatten() for a in np.split(pred, pred.shape[-1], axis=-1)]

    coords = np.arange(pred.shape[0] * pred.shape[1])
    y = (y_f + coords // pred.shape[0]) / pred.shape[0]
    x = (x_f + coords % pred.shape[1]) / pred.shape[1]

    boxes = np.stack([y, x, height, width, score], axis=-1)
    boxes = boxes[np.where(boxes[..., -1] >= SCORE_THRESHOLD)]
    dimensions = draw_outputs(original_image, boxes)

    
    return dimensions
'''

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

'''  