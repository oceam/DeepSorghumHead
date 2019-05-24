#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 17:06:33 2018

@author: sam1993
"""

import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import csv
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
#model_path = '/media/sam1993/prime/RCNN_keras/keras-retinanet-master/keras_retinanet/bin/inf_models/all_samples/resnet50_csv_16_inf.h5'
#model_path = '/media/sam1993/prime/RCNN_keras/keras-retinanet-master/keras_retinanet/bin/inf_models/50samp_resnet50_csv_50_inf.h5'
model_path = '/media/sam1993/prime/RCNN_keras/keras-retinanet-master/keras_retinanet/bin/inf_models/50samp_resnet50_csv_01_mAP_0.7797_inf.h5'
#model_path = '/media/sam1993/prime/RCNN_keras/keras-retinanet-master/keras_retinanet/bin/inf_models/all_samples/resnet152_csv_16_inf.h5'

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')
#model = models.load_model(model_path, backbone_name='resnet101')
#model = models.load_model(model_path, backbone_name='resnet152')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.load_model(model_path, backbone_name='resnet50', convert_model=True)

print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'SH'}

# load image
#image = read_image_bgr('/media/sam1993/prime/sorghum/Dots_Label/52Croppedimages/Data_croped_test/C1-R27-G23-DSC01480.tif')
image = read_image_bgr('/media/sam1993/prime/RCNN_keras/keras-retinanet-master/keras_retinanet/bin/inf_models/T2/1_ORIGINAL.tif')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

# Save Bounding Box co-ordinates
boxes_A = np.squeeze(boxes, axis=0)
np.savetxt("/media/sam1993/prime/RCNN_keras/keras-retinanet-master/keras_retinanet/bin/bounding_boxes/output.csv", boxes_A, delimiter=",")

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5: # IoU (Intersection over Union)
        break
        
    color = label_color(label)
    
    b = box.astype(int)
    draw_box(draw, b, color=color)
    
    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)
    
fig = plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()
fig.savefig('/media/sam1993/prime/RCNN_keras/keras-retinanet-master/keras_retinanet/bin/inf_models/T2/allsamp_resnet152_csv_16_C21-R10-G471-DSC00815.png', bbox_inches='tight', pad_inches=0)

print('SH Head Count: ', boxes.shape[1])
