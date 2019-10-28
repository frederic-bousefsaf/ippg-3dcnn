## UNCOMMENTING THESE TWO LINES WILL FORCE KERAS/TF TO RUN ON CPU
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import model_from_json

from tensorflow.python.keras._impl.keras.utils import np_utils
# from tensorflow.python.keras.utils import np_utils

import numpy as np
import scipy.io
import scipy.stats as sp
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import cv2
import os
from copy import copy


root = tk.Tk()
root.withdraw()
DIR_TESTING_DATA = filedialog.askdirectory(initialdir='D:/images/')
DIR_SAVE = 'D:/SAVE_DIR/'

if(DIR_TESTING_DATA==''):
    exit()

list_dir = os.listdir(DIR_TESTING_DATA)
root.destroy()
del root


# 0. CONSTANT
USE_RANDOM_PIXEL_LOCATION = 1
NB_TOTAL_IMAGES_PER_VIDEO_TESTING = len(list_dir)

# Load an image to get the frame dimensions
temp = cv2.imread(DIR_TESTING_DATA + '/' + list_dir[0], cv2.IMREAD_ANYCOLOR)
IMAGE_WIDTH = temp.shape[1]
IMAGE_HEIGHT = temp.shape[0]

if (len(temp.shape)<3):
    IMAGE_CHANNELS = 1
else:
    IMAGE_CHANNELS = temp.shape[2]



# 1. LOAD PRETRAINED MODEL
model = model_from_json(open('model_conv3D.json').read())
model.load_weights('weights_conv3D.h5')
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# define the frequencies // output dimension (number of classes used during training)
freq_BPM = np.linspace(55, 240, num=model.output_shape[1]-1)
freq_BPM = np.append(freq_BPM, -1)     # noise class

# define patch size and number of images per video (directly from the model information)
PATCH_WIDTH = model.input_shape[2]
PATCH_HEIGHT = model.input_shape[3]
NB_SELECTED_IMAGES_PER_VIDEO = model.input_shape[1]



# 2. LOAD DATA
imgs = np.zeros(shape=(NB_TOTAL_IMAGES_PER_VIDEO_TESTING, IMAGE_HEIGHT, IMAGE_WIDTH, 1))

# load images (imgs contains the whole video)
for j in range(NB_TOTAL_IMAGES_PER_VIDEO_TESTING):
    temp = cv2.imread(DIR_TESTING_DATA + '/' + list_dir[j], cv2.IMREAD_ANYCOLOR)

    if (IMAGE_CHANNELS==3):
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)/255
        temp = temp[:,:,1]      # only the G component is currently used
    else:
        temp = temp / 255

    imgs[j] = np.expand_dims(temp, 2)



# prepare data (xtest contains excerpts of NB_SELECTED_IMAGES_PER_VIDEO frames) and predict for each patch 
p_map = np.zeros(shape=(NB_TOTAL_IMAGES_PER_VIDEO_TESTING - NB_SELECTED_IMAGES_PER_VIDEO + 1, IMAGE_HEIGHT-PATCH_HEIGHT+1, IMAGE_WIDTH-PATCH_WIDTH+1, len(freq_BPM)))

for m in range(0, IMAGE_WIDTH-PATCH_WIDTH+1):
    for n in range(0, IMAGE_HEIGHT-PATCH_HEIGHT+1):
        patch = copy(imgs[:,n:n+PATCH_HEIGHT,m:m+PATCH_WIDTH,:])
        
        # randomize pixel locations
        if (USE_RANDOM_PIXEL_LOCATION==1):
            for j in range(NB_TOTAL_IMAGES_PER_VIDEO_TESTING):
                temp = np.reshape(patch[j,:,:,0], (PATCH_HEIGHT * PATCH_WIDTH))
                np.random.shuffle(temp)
                patch[j] = np.expand_dims(np.reshape(temp, (PATCH_HEIGHT, PATCH_WIDTH)), 2)
        
        for j in range(NB_TOTAL_IMAGES_PER_VIDEO_TESTING - NB_SELECTED_IMAGES_PER_VIDEO + 1):
            xtest = patch[j:j + NB_SELECTED_IMAGES_PER_VIDEO] - np.mean(patch[j:j + NB_SELECTED_IMAGES_PER_VIDEO])
            h = model.predict(np.expand_dims(xtest, 0))
            p_map[j,n,m] = h

    print('progress: row ' + str(m+1) + ' over ' + str(IMAGE_WIDTH+1-PATCH_WIDTH))


# save results
if (not os.path.isdir(DIR_SAVE)):
    os.mkdir(DIR_SAVE)


data = {}
data['map_p'] = p_map
scipy.io.savemat(DIR_SAVE + 'results.mat', data)
