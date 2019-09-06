## UNCOMMENTING THESE TWO LINES WILL FORCE KERAS/TF TO RUN ON CPU
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.layers import ZeroPadding3D, Dense, Activation,Conv3D,MaxPooling3D,AveragePooling3D,Flatten,Dropout
#from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras._impl.keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import scipy.io
import generate_trend

# np.random.seed(8)


# CONSTANTS
NB_VIDEOS_BY_CLASS_TRAIN = 200
NB_VIDEOS_BY_CLASS_TEST = 200

# Tendencies (linear, 2nd order, 3rd order)
TENDANCIES_MIN = (-3,-1,-1)
TENDANCIES_MAX = (3,1,1)
TENDANCIES_ORDER = (1,2,3)

LENGTH_VIDEO = 60
IMAGE_WIDTH = 25
IMAGE_HEIGHT = 25
IMAGE_CHANNELS = 1

SAMPLING = 1 / 30
t = np.linspace(0, LENGTH_VIDEO * SAMPLING - SAMPLING, LENGTH_VIDEO)

# coefficients for the fitted-ppg method
a0 = 0.440240602542388
a1 = -0.334501803331783
b1 = -0.198990393984879
a2 = -0.050159136439220
b2 = 0.099347477830878
w = 2 * np.pi

HEART_RATES = np.linspace(55, 150, 39)
# HEART_RATES = np.linspace(55, 240, 75)
NB_CLASSES = len(HEART_RATES)

# prepare labels and label categories
labels = np.zeros(NB_CLASSES + 1)

for i in range(NB_CLASSES + 1):
    labels[i] = i
labels_cat = np_utils.to_categorical(labels)


EPOCHS = 1000
CONTINUE_TRAINING = False
SAVE_ALL_MODELS = False
train_loss = []
val_loss = []
train_acc = []
val_acc = []


# 1.  DEFINE OR LOAD MODEL / WEIGHTS
if (CONTINUE_TRAINING == False):
    init_batch_nb = 0

    model = Sequential()

    model.add(Conv3D(filters=32, kernel_size=(58,20,20), input_shape=(LENGTH_VIDEO, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(NB_CLASSES + 1, activation='softmax'))

else:
    # load model
    model = model_from_json(open('../../model_conv3D.json').read())
    model.load_weights('../../weights_conv3D.h5')
    
    # load statistics
    dummy = np.loadtxt('../../statistics_loss_acc.txt')
    init_batch_nb = dummy.shape[0]
    train_loss = dummy[:,0].tolist()
    train_acc = dummy[:,1].tolist()
    val_loss = dummy[:,2].tolist()
    val_acc = dummy[:,3].tolist()



model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

data = {}

# 2.  GENERATE TEST DATA
xtest = np.zeros(shape=((NB_CLASSES + 1) * NB_VIDEOS_BY_CLASS_TEST, LENGTH_VIDEO, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
ytest = np.zeros(shape=((NB_CLASSES + 1) * NB_VIDEOS_BY_CLASS_TEST, NB_CLASSES + 1))

c = 0

# for each frequency
for i_freq in range(len(HEART_RATES)):

    for i_videos in range(NB_VIDEOS_BY_CLASS_TEST):

        t2 = t + (np.random.randint(low=0, high=33) * SAMPLING)   # phase. 33 corresponds to a full phase shift for HR=55 bpm
        signal = a0 + a1 * np.cos(t2 * w * HEART_RATES[i_freq] / 60) + b1 * np.sin(t2 * w * HEART_RATES[i_freq] / 60) + a2 * np.cos(2 * t2 * w * HEART_RATES[i_freq] / 60) + b2 * np.sin(2 * t2 * w * HEART_RATES[i_freq] / 60)
        signal = signal - np.min(signal)
        signal = signal / np.max(signal)

        r = np.random.randint(low=0, high=len(TENDANCIES_MAX))      # high value is not comprised (exclusive)
        trend = generate_trend.generate_trend(len(t), TENDANCIES_ORDER[r], 0, np.random.uniform(low=TENDANCIES_MIN[r], high=TENDANCIES_MAX[r]), np.random.randint(low=0, high=2))

        signal = np.expand_dims(signal + trend, 1)
        signal = signal - np.min(signal)

        img = np.tile(signal, (IMAGE_WIDTH, 1, IMAGE_HEIGHT))
        img = np.transpose(img, axes=(0,2,1))

        img = img / (IMAGE_HEIGHT * IMAGE_WIDTH)
        
        amplitude = np.random.uniform(low=1.5, high=4)
        noise_energy = amplitude * 0.25 * np.random.uniform(low=1, high=10) / 100

        for j in range(0, LENGTH_VIDEO):
            temp = 255 * ((amplitude * img[:,:,j]) + np.random.normal(size=(IMAGE_HEIGHT, IMAGE_WIDTH), loc=0.5, scale=0.25) * noise_energy)
            temp[temp < 0] = 0 
            xtest[c,j,:,:,0] = temp.astype('uint8') / 255.0

        xtest[c] = xtest[c] - np.mean(xtest[c])
        ytest[c] = labels_cat[i_freq]

        c = c + 1
        #data['new'] = xtest[0:c-1,:,:,:,0]
        #scipy.io.savemat('D:/Users/bousefsa1/Desktop/new.mat', data)

# constant image noise (gaussian distribution)
for i_videos in range(NB_VIDEOS_BY_CLASS_TEST):
    r = np.random.randint(low=0, high=len(TENDANCIES_MAX))      # high value is not comprised (exclusive)
    trend = generate_trend.generate_trend(len(t), TENDANCIES_ORDER[r], 0, np.random.uniform(low=TENDANCIES_MIN[r], high=TENDANCIES_MAX[r]), np.random.randint(low=0, high=2))

    # add a tendancy on noise
    signal = np.expand_dims(trend, 1)
    img = np.tile(signal, (IMAGE_WIDTH, 1, IMAGE_HEIGHT)) / (IMAGE_HEIGHT * IMAGE_WIDTH)
    img = np.expand_dims(np.transpose(img, axes=(1,0,2)), 3)

    xtest[c] = np.expand_dims(np.random.normal(size=(LENGTH_VIDEO, IMAGE_HEIGHT, IMAGE_WIDTH)) / 50, 3) + img
    xtest[c] = xtest[c] - np.mean(xtest[c])
    ytest[c] = labels_cat[NB_CLASSES]
    c = c + 1

print('Test data generation done')

## Load test data from matlab file
#tt = scipy.io.loadmat('D:/Users/bousefsa1/Desktop/matlab.mat')
#xtest = np.expand_dims(tt['x'],5)
#for i_videos in range(xtest.shape[0]):
#    xtest[i_videos] = xtest[i_videos] - np.mean(xtest[i_videos])

#ytest = labels_cat[tt['y']]
#ytest = ytest[0]



# 3.  GENERATE TRAINING DATA AND TRAIN ON BATCH
xtrain = np.zeros(shape=((NB_CLASSES + 1) * NB_VIDEOS_BY_CLASS_TRAIN, LENGTH_VIDEO, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
ytrain = np.zeros(shape=((NB_CLASSES + 1) * NB_VIDEOS_BY_CLASS_TRAIN, NB_CLASSES + 1))

c = 0

for batch_nb in range(init_batch_nb, EPOCHS):
    for i_freq in range(len(HEART_RATES)):

        for i_videos in range(NB_VIDEOS_BY_CLASS_TRAIN):

            t2 = t + (np.random.randint(low=0, high=33) * SAMPLING)   # phase
            signal = a0 + a1 * np.cos(t2 * w * HEART_RATES[i_freq] / 60) + b1 * np.sin(t2 * w * HEART_RATES[i_freq] / 60) + a2 * np.cos(2 * t2 * w * HEART_RATES[i_freq] / 60) + b2 * np.sin(2 * t2 * w * HEART_RATES[i_freq] / 60)
            signal = signal - np.min(signal)
            signal = signal / np.max(signal)

            r = np.random.randint(low=0, high=len(TENDANCIES_MAX))      # high value is not comprised (exclusive)
            trend = generate_trend.generate_trend(len(t), TENDANCIES_ORDER[r], 0, np.random.uniform(low=TENDANCIES_MIN[r], high=TENDANCIES_MAX[r]), np.random.randint(low=0, high=2))

            signal = np.expand_dims(signal + trend, 1)
            signal = signal - np.min(signal)

            img = np.tile(signal, (IMAGE_WIDTH, 1, IMAGE_HEIGHT))
            img = np.transpose(img, axes=(0,2,1))

            img = img / (IMAGE_HEIGHT * IMAGE_WIDTH)
        
            amplitude = np.random.uniform(low=1.5, high=4)
            noise_energy = amplitude * 0.25 * np.random.uniform(low=1, high=10) / 100

            for j in range(0, LENGTH_VIDEO):
                temp = 255 * ((amplitude * img[:,:,j]) + np.random.normal(size=(IMAGE_HEIGHT, IMAGE_WIDTH), loc=0.5, scale=0.25) * noise_energy)
                temp[temp < 0] = 0 
                xtrain[c,j,:,:,0] = temp.astype('uint8') / 255.0

            xtrain[c] = xtrain[c] - np.mean(xtrain[c])
            ytrain[c] = labels_cat[i_freq]

            c = c + 1


    # constant image noise (gaussian distribution)
    for i_videos in range(NB_VIDEOS_BY_CLASS_TRAIN):
        r = np.random.randint(low=0, high=len(TENDANCIES_MAX))      # high value is not comprised (exclusive)
        trend = generate_trend.generate_trend(len(t), TENDANCIES_ORDER[r], 0, np.random.uniform(low=TENDANCIES_MIN[r], high=TENDANCIES_MAX[r]), np.random.randint(low=0, high=2))

        # add a tendancy on noise
        signal = np.expand_dims(trend, 1)
        img = np.tile(signal, (IMAGE_WIDTH, 1, IMAGE_HEIGHT)) / (IMAGE_HEIGHT * IMAGE_WIDTH)
        img = np.expand_dims(np.transpose(img, axes=(1,0,2)), 3)

        xtrain[c] = np.expand_dims(np.random.normal(size=(LENGTH_VIDEO, IMAGE_HEIGHT, IMAGE_WIDTH)) / 50, 3) + img
        xtrain[c] = xtrain[c] - np.mean(xtrain[c])
        ytrain[c] = labels_cat[NB_CLASSES]
        c = c + 1

    print('Train data (batch) generation done. Starting training...')


    history = model.train_on_batch(xtrain, ytrain)
    train_loss.append(history[0])
    train_acc.append(history[1])

    history = model.evaluate(xtest, ytest, verbose=2)
    

    # A. Save the model only if the accuracy is greater than before
    if (SAVE_ALL_MODELS==False):
        if (batch_nb > 0):
            f1 = open('../../statistics_loss_acc.txt', 'a')

            # save model and weights if val_acc is greater than before
            if (history[1] > np.max(val_acc)):
                model.save_weights('../../models/weights_conv3D.h5', overwrite=True)   # save (trained) weights
                print('A new model has been saved!\n')
        else:
            if not os.path.exists('../../models'):
                os.makedirs('../../models')

            f1 = open('../../models/statistics_loss_acc.txt', 'w')
            model_json = model.to_json()
            open('../../models/model_conv3D.json', 'w').write(model_json)        # save model architecture

    
    # B. Save the model every iteration
    else:
        if (batch_nb > 0):
            f1 = open('../../models/statistics_loss_acc.txt', 'a')
       
        else:
            if not os.path.exists('../../models'):
                os.makedirs('../../models')

            f1 = open('../../models/statistics_loss_acc.txt', 'w')
            model_json = model.to_json()
            open('../../models/model_conv3D.json', 'w').write(model_json)                       # save model architecture

        model.save_weights('../../models/weights_conv3D_%04d.h5' % batch_nb, overwrite=True)    # save (trained) weights

    
    
    val_loss.append(history[0])
    val_acc.append(history[1])

    print('training: ' + str(batch_nb + 1) + '/' + str(EPOCHS) + ' done')
    print('training: loss=' + str(train_loss[batch_nb]) + ' acc=' + str(train_acc[batch_nb]))
    print('validation: loss=' + str(val_loss[batch_nb]) + ' acc=' + str(val_acc[batch_nb]) + '\n')


    # save learning state informations
    f1.write(str(train_loss[batch_nb]) + '\t' + str(train_acc[batch_nb]) + '\t' + str(val_loss[batch_nb]) + '\t' + str(val_acc[batch_nb]) + '\n')
    f1.close()
   
    c = 0


# plot history for accuracy
plt.subplot(211)
plt.plot(train_acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# plot history for loss
plt.subplot(212)
plt.plot(train_loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.tight_layout()
plt.show()
