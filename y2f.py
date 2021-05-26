# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:35:16 2021

@author: ghior
"""

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

from Read_SpectralytoF import *   # data set build
from recoverynetx2 import *     # Net build
from Metrics import *


# To run in cpu
os.environ["CUDA_VISIBLE_DEVICES"]= '-1'

#----------------------------- directory of the spectral data set -----------------                        # for windows

''' data set 1 '''
PATH = r'C:\Users\ghior\Documents\MATLAB\DeepApproachHtransformation\Data\yi2fTrain'
''' data set 2 '''
#PATH = r'C:\Users\ghior\Documents\MATLAB\DeepApproachHtransformation\Data\MedidasDeep\aradTrainF'

# parameters of the net
#BATCH_SIZE = 2; IMG_WIDTH = 140; IMG_HEIGHT = 128; L_bands    = 8; L_imput    = 8; split_v = 0.80
BATCH_SIZE = 2; IMG_WIDTH = 128; IMG_HEIGHT = 128; L_bands    = 13; L_imput    = 13; split_v = 0.80


#epocas = 100
#epocas = 250
#epocas = 500
epocas = 1

#lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,decay_steps=100,decay_rate=0.9)
#optimizad = keras.optimizers.SGD(learning_rate=lr_schedule)
optimizad = tf.keras.optimizers.Adam(learning_rate=1e-4, amsgrad=False)

test_dataset,train_dataset=Build_data_set(IMG_HEIGHT,IMG_WIDTH,L_bands,L_imput,BATCH_SIZE,PATH,split_v)  # build the DataStore from Read_Spectral.py
    
#-------------Net_model----------------------------------------------------------------
preWe = r'modelosEntrenados/UNetYtoFep750.h5'
modely1 = UNetL_test(input_size=(IMG_HEIGHT,IMG_WIDTH,L_bands),pretrained_weights=preWe)
    
    
modely1.compile(optimizer=optimizad, loss='mean_squared_error',metrics=[psnr,'mse','mae',SSIM])

#model.load_weights("UNetAdaptx2.h5")


history = modely1.fit_generator(train_dataset, epochs=epocas,shuffle=True,validation_data=(test_dataset))

#.------------ seee the accuracy---------------
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('REDytoy1/y2ftrainEp'+str(epocas)+'.png')
plt.savefig('REDytoy1/y2ftrainEp'+str(epocas)+'.eps', dpi=500)
plt.show()

modely1.save_weights('modelosEntrenados/UNetYtoFep'+str(epocas)+'.h5')
  
  
test_loss = modely1.evaluate(test_dataset,  verbose=2)
print('.....testing loss y to y1: ' +str(test_loss))


#
if True:
    
    ''' data loading '''
    
    saved = r'C:\Users\ghior\Documents\MATLAB\DeepApproachHtransformation\Data\yi2f\y2f_'
    toSave = r'C:\Users\ghior\Documents\MATLAB\DeepApproachHtransformation\DeepLearning\10teamH\outY4y'
    samples = 150
    hy_list = []
    hy1_list = []
    hy2_list = []
    for i in np.arange(samples):
        file = saved + str((i)) + '.mat'
        temp_m = loadmat(file)
 
        inImaBCC_00 = temp_m['Ibcc']
        Ref_img=np.expand_dims(inImaBCC_00,0)
        outImaBCC_00 = modely1.predict(Ref_img,batch_size=1)
        
        Resul=outImaBCC_00[0,:,:,:]
                
        out_path = toSave + '\outaF_' + str(i) + '.mat'
        
        scipy.io.savemat(out_path, dict(y_pred=Resul))
        






