# -*- coding: utf-8 -*-

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

from retina.utils import visualize_boxes

MODEL_PATH = 'snapshots/resnet50_full.h5'
IMAGE_NAME = 'test82.png'
IMAGE_PATH = 'samples/JPEGImages/'
SAVE_PATH = 'samples/samples/'

def load_inference_model(model_path=os.path.join('snapshots', 'resnet.h5')):
    model = models.load_model(model_path, backbone_name='resnet50',)
    model = models.convert_model(model)
    #model.summary()
    return model

def post_process(boxes, original_img, preprocessed_img):
    # post-processing
    h, w, _ = preprocessed_img.shape
    h2, w2, _ = original_img.shape
    boxes[:, :, 0] = boxes[:, :, 0] / w * w2
    boxes[:, :, 2] = boxes[:, :, 2] / w * w2
    boxes[:, :, 1] = boxes[:, :, 1] / h * h2
    boxes[:, :, 3] = boxes[:, :, 3] / h * h2
    return boxes

def rotate(img,angle=90):

    rows = img.shape[0]
    cols = img.shape[1]

    img_center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(img_center, angle, 1)
    rotated_image = cv2.warpAffine(img, M, (cols, rows))

    return rotated_image
def main():
    scores = [0]
    angle = 0
    iter = 1
    start = time.time()
    while((np.mean(scores[:3]) < 0.7) and (angle < 180) and (np.mean(scores[:2]) < 0.65) and iter <= 10):

        model = load_inference_model(MODEL_PATH)
        
        # load image
        image = read_image_bgr(IMAGE_PATH + IMAGE_NAME)
        
        #Rotate
        if(iter != 1):
            angle = angle + 20

        image = rotate(image,angle)

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        #rotate
        
        
        # preprocess image for network
        image = preprocess_image(image)
        image, _ = resize_image(image, 416, 448)
        
        # process image
        #start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("iteration: ",str(iter))
        print("processing time: ", time.time() - start)
        
        boxes = post_process(boxes, draw, image)
        labels = labels[0]
        #print(labels)
        scores = scores[0]
        #print(scores)
        boxes = boxes[0]
        
        if(iter == 1):
            angle = -40
        iter = iter + 1

    #print(labels)
    #print(scores)    
    visualize_boxes(draw, boxes, labels, scores, class_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    # 5. plot    
    plt.imshow(draw)
    plt.imsave(SAVE_PATH + IMAGE_NAME,draw)
    plt.show()
    

if __name__ == '__main__':
    
    main()

