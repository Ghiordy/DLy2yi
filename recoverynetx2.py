
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from IPython import display
from IPython.display import clear_output

from os import listdir
from os.path import isfile, join

import numpy as np
import keras
import scipy.io
from scipy.io import loadmat
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

# important part load the psf layer

def UNetL(pretrained_weights=None, input_size=(256, 256, 12)):

    inputs = Input(input_size)
    L = input_size[2];
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)

    up5 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv4))
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv7)

    final = Conv2D(L, 1)(conv7)


    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def UNetL2(pretrained_weights=None, input_size=(256, 256, 12)):

    inputs = Input(input_size)
    L = input_size[2];
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(1, 1))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)

    up5 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(1, 1))(conv4))
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv7)

    final = Conv2D(L, 1)(conv7)


    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def UNetL1(pretrained_weights=None, input_size=(256, 256, 12)):

    inputs = Input(input_size)
    L = input_size[2];
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    #pool1 = MaxPooling2D(pool_size=(1, 1))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    #pool2 = MaxPooling2D(pool_size=(1, 1))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    #pool3 = MaxPooling2D(pool_size=(1, 1))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)

    #up5 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #    UpSampling2D(size=(1, 1))(conv4))
    #merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    #up6 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #    UpSampling2D(size=(1, 1))(conv5))
    #merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    #up7 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #    UpSampling2D(size=(1, 1))(conv6))
    #merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv7)

    final = Conv2D(L, 1)(conv7)


    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def UNetL_test(pretrained_weights=None,input_size=(512,512,1),L=32,FLAGS=None):
    inputs=Input(input_size)
    L_2=2 * L;
    L_3=3 * L;
    L_4=4 * L;

    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(inputs)
    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv2)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3=MaxPooling2D(pool_size=(2,2))(conv3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)

    up5=Conv2D(L_3,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv4))
    merge5=concatenate([conv3,up5],axis=3)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge5)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv5)

    up6=Conv2D(L_2,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv5))
    merge6=concatenate([conv2,up6],axis=3)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge6)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv6)

    up7=Conv2D(L,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv6))
    merge7=concatenate([conv1,up7],axis=3)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge7)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal',)(conv7)

    final=Conv2D(input_size[2],3,activation='relu',padding='same',kernel_initializer='he_normal',)(conv7)



    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def UNetL_testYiY(pretrained_weights=None,input_size=(512,512,1),L=32,FLAGS=None):
    inputs=Input(input_size)
    L_2=2 * L;
    L_3=3 * L;
    L_4=4 * L;
    #pool0=MaxPooling2D(pool_size=(2,2))(inputs)
    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(inputs)
    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv2)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3=MaxPooling2D(pool_size=(2,2))(conv3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)

    up5=Conv2D(L_3,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv4))
    merge5=concatenate([conv3,up5],axis=3)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge5)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv5)

    up6=Conv2D(L_2,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv5))
    merge6=concatenate([conv2,up6],axis=3)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge6)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv6)

    up7=Conv2D(L,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv6))
    #merge7=concatenate([conv1,up7],axis=3)
    #conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge7)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(up7)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal',)(conv7)

    final=Conv2D(input_size[2],3,activation='relu',padding='same',kernel_initializer='he_normal',)(conv7)



    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def UNetL_AE(pretrained_weights=None,input_size=(512,512,1),L=32,FLAGS=None):
    inputs=Input(input_size)
    L_2=2 * L;
    L_3=3 * L;
    L_4=4 * L;

    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(inputs)
    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv2)
    pool2=MaxPooling2D(pool_size=(2,1))(conv2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3=MaxPooling2D(pool_size=(2,1))(conv3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)

    up5=Conv2D(L_3,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,1))(conv4))
    merge5=concatenate([conv3,up5],axis=3)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge5)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv5)

    up6=Conv2D(L_2,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,1))(conv5))
    merge6=concatenate([conv2,up6],axis=3)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge6)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv6)

    up7=Conv2D(L,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv6))
    merge7=concatenate([conv1,up7],axis=3)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge7)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal',)(conv7)

    final=Conv2D(input_size[2],3,activation='relu',padding='same',kernel_initializer='he_normal',)(conv7)



    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


## Autoencoder tf recomended ##############################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
#from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


def Denoise(pretrained_weights=None,input_size=(512,512,1),L=8,FLAGS=None):
    inputs=Input(input_size)
    #L_2=2 * L;
    L_2 = 35;
    #L_3=3 * L;
    L_3 = 128;
    #L_4=4 * L;
    L_4= 140;

    x = layers.Conv2D(L, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(inputs)
    #x = layers.Conv2D(L, (L, L), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L, (L, L), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L, (L, L), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(L_2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L_2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L_2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L_2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    #x = layers.Conv2D(L_3, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L_3, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L_3, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L_3, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #encoded = layers.MaxPooling2D(pool_size=(2, 1))(x)

    # At this point the representation is (7, 7, 32)

    #x = layers.Conv2D(L_4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(encoded)
    x = layers.Conv2D(L_4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L_4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L_4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.UpSampling2D((2, 1))(x)
    #x = layers.Conv2D(L_3, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L_3, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L_3, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L_3, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(L_2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L_2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L_2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L_2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(L, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    #x = layers.Conv2D(L, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(x)
    decoded = layers.Conv2D(L, (3, 3), activation='sigmoid', padding='same',kernel_initializer='he_normal')(x)

    model = Model(inputs, decoded)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    
    return model
