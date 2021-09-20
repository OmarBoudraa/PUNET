# This is the classifier code which trains a model to generate
# the PUNET of a word image.

# Author: Omar BOUDRAA <o_boudraa@esi.dz>

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

# The below code ensures that GPU memory is dynamically allocated
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

from keras.utils import multi_gpu_model # If we want to use multiple GPUs

from keras.models import Sequential, model_from_json, Model
from keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
                    LeakyReLU, Activation, Input, UpSampling2D, concatenate)
from keras.optimizers import SGD
from keras import losses
from keras.callbacks import TensorBoard

from load_data import load_data
from save_load_weight import *
from evaluate_punet import *

# Thanks to https://github.com/yhenon/keras-spp for the SPP Layer
from spp.SpatialPyramidPooling import SpatialPyramidPooling
from datetime import datetime

JUMP = 16 # Number of Epochs after we Save Model
GPUS = 2 # Number of GPU we want to use for Train & Test

def create_model(pretrained_weights = None,input_size = (64, 160,1)):
  """This module creates an Instance of the Sequential Class in Keras.

  Args:
    None.

  Return:
    model: Instance of the Sequential Class
  """
  time_start = datetime.now()
  input_shape=(None, None, 1)
  #inputs = Input(shape=input_size[1:])
  #inputs = Input(shape=(64,160,1))
  inputs = Input(input_size)

  conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
  conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
  conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
  conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
  conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
  drop4 = Dropout(0.5)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

  conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
  conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
  drop5 = Dropout(0.5)(conv5)

  up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(drop5))
  merge6 = concatenate([drop4, up6], axis=3)
  conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
  conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

  up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv6))
  merge7 = concatenate([conv3, up7], axis=3)
  conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
  conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

  up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv7))
  merge8 = concatenate([conv2, up8], axis=3)
  conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
  conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

  up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv8))
  merge9 = concatenate([conv1, up9], axis=3)
  conv9 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
  conv9 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
  #conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
  #conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

  conv9 = SpatialPyramidPooling([1, 2, 4])(conv9)
  conv9 = Dense(4096, activation='relu')(conv9)
  conv9 = Dropout(0.5)(conv9)
  conv9 = Dense(4096, activation='relu')(conv9)
  conv9 = Dropout(0.5)(conv9)
  conv10 = Dense(604, activation='sigmoid')(conv9)

  model = Model(inputs=inputs, outputs=conv10)


  model = multi_gpu_model(model, gpus=GPUS)

  loss = losses.binary_crossentropy
  optimizer = SGD(lr=1e-4, momentum=.9, decay=5e-5)
  model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', 'mae'])
  model.summary()
  print ("Time taken to create model: ", datetime.now()-time_start)

  # model.summary()

  if (pretrained_weights):
    model.load_weights(pretrained_weights)

  return model


def trainer(model, x_train, y_train, x_valid, y_valid, initial_epoch=0):
  """This trains the model partially and
  returns the partially trained weights

  Args:
    model: Instance of the Sequential Class storing the Neural Network
    x_train: Numpy storing the training Images
    y_train: Numpy storing the PUNET Label of the training Images
    x_valid: Numpy storing the Validation Images
    y_valid: Numpy storing the PUNET Labels of Validation Data
    initial_epoch: Starting Epoch of the partial Train (Default: 0)

  Returns:
    model: Instance of the Sequential Class having partially trained model
  """
  #x_valid = x_valid.reshape( (1,50) )
  tnsbrd = TensorBoard(log_dir='./logs')
  model.fit(x_train,
            y_train,
            batch_size=10,
            callbacks=[tnsbrd],
            epochs=initial_epoch+JUMP,
            initial_epoch=initial_epoch,
            validation_data=(x_valid, y_valid))
  return model


def train(initial_epoch=0):
  """This is the main driver function which first partialy trains the model.
  Then it passes to the weight Saving & Loading modules.

  Args:
    initial_epoch: Integer. Provides the starting point for Training.

  Returns:
    None.
  """
  time_start = datetime.now()
  model = create_model()
  if initial_epoch: # If you are not starting from begining
    model = load_model_weight(model)
  data = load_data() 

  x_train = data[0]
  y_train = data[1]
  x_valid = data[3]
  y_valid = data[4]
  x_test = data[6]
  y_test = data[7]
  test_transcripts = data[8]
  for i in range(initial_epoch, 12, JUMP):
    model = trainer(model, x_train, y_train, x_valid, y_valid, initial_epoch=i)
    save_model_weight(model) # Saves the model
    map(model, x_test, y_test, test_transcripts) # Calculates the MAP of the model
  print ("Time taken to train the entire model: ", datetime.now()-time_start)
