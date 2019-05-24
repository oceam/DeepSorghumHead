#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:47:37 2018

@author: sghosal
"""

from __future__ import print_function

import os
import keras

from keras import Model
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr

import numpy as np
import keras.backend as K

import matplotlib.cm as cm
import glob
import scipy
import scipy.io
import cv2
import pylab as pl

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

def target_category_loss(x, category_index, nb_classes):
    
    return K.dot(x, K.one_hot(tuple(category_index), nb_classes))  

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def target_category_loss_output_shape(input_shape):
    return input_shape

#########################################
### Developing and Training the model ###
#########################################

model_path = '#### ENTER YOUR TRAINED MODEL PATH ####'

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
#model = models.load_model(model_path, backbone_name='resnet50', convert_model=True)

print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'SH'}

print("Created model and loaded weights from file")

print("Loading data........") 

# utility functions
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Filter Visualization function
def filt_vis(input_model, image, input_layer_index):

    model = input_model
    
intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[input_layer_index].output)
    intermediate_output = intermediate_layer_model.predict(image)

    intermediate_output = np.moveaxis(intermediate_output, -1, 1) # IMPORTANT STEP

    return intermediate_output

def get_activations(model, layer, X):

    get_activations = K.function([model.layers[0].input], model.layers[layer].output)  

    activations = get_activations([X])[0]
    
    return activations

print ('generating the Feature Maps........')

counter = 0
layer_index = 3 # 0 to 215

for filename in glob.glob('#### ENTER YOUR IMAGE FOLDER PATH CONTAINING MULTIPLE TEST IMAGES ####'):
    image = read_image_bgr(filename)
    image = np.expand_dims(image, axis=0)
    
    processed_input = image

    filt  = filt_vis(model, processed_input, layer_index)

    filt = np.squeeze(filt)

    T1 = int(filt.shape[1:2][0])

    T2 = len(filt)*T1

    original = np.uint8(255 * processed_input[1:2])

    file_path = '#### ENTER YOUR FEATURE MAP OUPUT SAVE PATH ####'
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    scipy.io.savemat('#### ENTER YOUR ACTIVATION OUTPUT SAVE PATH (.mat file) ####'%(layer_index, filename[34:-4], counter), mdict={'Filters_%s'%(counter): filt}) ###############
    
    counter1 = 1
    for j in range(len(filt)):
        
        filt_rgb = filt[j]
        filt_rgb = np.uint8(255 * filt_rgb)
        filt_rgb = cv2.applyColorMap(np.uint8(255 * filt_rgb), cv2.COLORMAP_JET)
        
        cv2.imwrite("#### ENTER YOUR FEATURE MAP OUPUT SAVE PATH ####"%(layer_index, filename[34:-4], counter, j), filt_rgb) ##############
        counter1 += 1   
    
    print('######################################################')
    print('IMAGE NUMBER DONE: ', counter)
    print('######################################################')
    counter += 1
