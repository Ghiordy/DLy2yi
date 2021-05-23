#  Structure Deep Learning project by HDSP group
# Any information contacted with jorge.bacca1@correo.uis.edu.co


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
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


from Main_ytoy1 import UNetYtoY1
from Main_ytoy2 import UNetYtoY2
from Main_ytoy1 import retUNetYtoY1
from Main_ytoy2 import retUNetYtoY2

# To run in cpu
os.environ["CUDA_VISIBLE_DEVICES"]= '-1'

#----------------------------- directory of the spectral data set -----------------                        # for windows

''' data set 1 '''
PATH = r'C:\Users\ghior\Documents\MATLAB\DeepApproachHtransformation\Data\MedidasDeep\data512imgTrain362'
''' data set 2 '''
#PATH = r'C:\Users\ghior\Documents\MATLAB\DeepApproachHtransformation\Data\MedidasDeep\aradTrainF'

# parameters of the net
BATCH_SIZE = 2; IMG_WIDTH = 140; IMG_HEIGHT = 128; L_bands    = 8; L_imput    = 8; split_v = 0.10

tra_ep = [64,128,256,512]
#epocas = 100
epocas = 16

if False:
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=8e-3,decay_steps=500,decay_rate=0.1)
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
    modely1 = UNetYtoY1(IMG_HEIGHT,IMG_WIDTH,L_bands,L_imput,BATCH_SIZE,PATH,split_v,epocas,optimizer)
    #modely2 = UNetYtoY2(IMG_HEIGHT,IMG_WIDTH,L_bands,L_imput,BATCH_SIZE,PATH,split_v,epocas,optimizer)

if False:
    optimizad = tf.keras.optimizers.Adam(learning_rate=1e-3, amsgrad=False)
    modely1 = UNetYtoY1(IMG_HEIGHT,IMG_WIDTH,L_bands,L_imput,BATCH_SIZE,PATH,split_v,epocas,optimizad)
    #modely2 = UNetYtoY2(IMG_HEIGHT,IMG_WIDTH,L_bands,L_imput,BATCH_SIZE,PATH,split_v,epocas,optimizad)
    
if True:
    #pretrained_weights1 = r'modelosEntrenados/UNetYtoY1ep1050A.h5'
    pretrained_weights1 = r'modelosEntrenados/UNetYtoY1ep632.h5'
    pretrained_weights2 = r'modelosEntrenados/UNetYtoY2ep1100.h5'
    optimizad = tf.keras.optimizers.Adam(learning_rate=5e-5, amsgrad=False)
    modely1 = retUNetYtoY1(IMG_HEIGHT,IMG_WIDTH,L_bands,L_imput,BATCH_SIZE,PATH,split_v,epocas,optimizad,pretrained_weights1)
    #modely2 = retUNetYtoY2(IMG_HEIGHT,IMG_WIDTH,L_bands,L_imput,BATCH_SIZE,PATH,split_v,epocas,optimizad,pretrained_weights2)


    
    

