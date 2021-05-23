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
#-------------------- This are main py------------------------

from Read_Spectraly1 import *   # data set build
from recoverynetx2 import *     # Net build
from Metrics import *


def UNetYtoY1(IMG_HEIGHT,IMG_WIDTH,L_bands,L_imput,BATCH_SIZE,PATH,split_v,epocas,optimizer):
    # To run in cpu
    os.environ["CUDA_VISIBLE_DEVICES"]= '-1'
    
    test_dataset,train_dataset=Build_data_set(IMG_HEIGHT,IMG_WIDTH,L_bands,L_imput,BATCH_SIZE,PATH,split_v)  # build the DataStore from Read_Spectral.py
    
    #-------------Net_model----------------------------------------------------------------
    #modely1 = UNetL2(input_size=(IMG_HEIGHT,IMG_WIDTH,L_bands))
    modely1 = UNetL_AE(input_size=(IMG_HEIGHT,IMG_WIDTH,L_bands))
    
    
    
    optimizad = optimizer
    
    
    modely1.compile(optimizer=optimizad, loss='mean_squared_error',run_eagerly=False,metrics=[psnr,'mse','mae',SSIM])
    
    #model.load_weights("UNetAdaptx2.h5")
    
    
    history = modely1.fit_generator(train_dataset, epochs=epocas,shuffle=True,validation_data=(test_dataset))
    
    #.------------ seee the accuracy---------------
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('REDytoy1/trainEp'+str(epocas)+'.png')
    plt.savefig('REDytoy1/trainEp'+str(epocas)+'.eps', dpi=500)
    plt.show()
    
    modely1.save_weights('modelosEntrenados/UNetYtoY1ep'+str(epocas)+'.h5')
  
  
    test_loss = modely1.evaluate(test_dataset,  verbose=2)
    print('.....testing loss y to y1: ' +str(test_loss))
    
    
    #
    if False:
        
        ''' data loading '''
        
        saved = r'C:\Users\ghior\Documents\MATLAB\DeepApproachHtransformation\Data\MedidasDeep\data512img\yT_'
        toSave = r'C:\Users\ghior\Documents\MATLAB\DeepApproachHtransformation\DeepLearning\10teamH\outY1'
        samples = 512
        hy_list = []
        hy1_list = []
        hy2_list = []
        for i in np.arange(samples):
            file = saved + str((i)) + '.mat'
            temp_m = loadmat(file)
     
            inImaBCC_00 = temp_m['re']
            Ref_img=np.expand_dims(inImaBCC_00,0)
            outImaBCC_00 = modely1.predict(Ref_img,batch_size=1)
            
            Resul=outImaBCC_00[0,:,:,:]
                    
            out_path = toSave + '\outa_' + str(i) + '.mat'
            
            scipy.io.savemat(out_path, dict(y_pred=Resul))
            
    return modely1
        
def retUNetYtoY1(IMG_HEIGHT,IMG_WIDTH,L_bands,L_imput,BATCH_SIZE,PATH,split_v,epocas,optimizer,weights):
    # To run in cpu
    os.environ["CUDA_VISIBLE_DEVICES"]= '-1'
    
    test_dataset,train_dataset=Build_data_set(IMG_HEIGHT,IMG_WIDTH,L_bands,L_imput,BATCH_SIZE,PATH,split_v)  # build the DataStore from Read_Spectral.py
    
    #-------------Net_model----------------------------------------------------------------
    modely1 = UNetL_AE(pretrained_weights=weights,input_size=(IMG_HEIGHT,IMG_WIDTH,L_bands))
      
    
    optimizad = optimizer
    
    
    modely1.compile(optimizer=optimizad, loss='mean_squared_error',run_eagerly=False,metrics=[psnr,'mse','mae',SSIM])
    
    #model.load_weights("UNetAdaptx2.h5")
    
    
    history = modely1.fit_generator(train_dataset, epochs=epocas,shuffle=True,validation_data=(test_dataset))
    
    #.------------ seee the accuracy---------------
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('REDytoy1/trainEp'+str(epocas)+'.png')
    plt.savefig('REDytoy1/trainEp'+str(epocas)+'.eps', dpi=500)
    plt.show()
    
    #modely1.save_weights('modelosEntrenados/UNetYtoY1ep'+str(epocas)+'.h5')
  
  
    test_loss = modely1.evaluate(test_dataset,  verbose=2)
    print('.....testing loss y to y1: ' +str(test_loss))
    
    probar = 'ARAD'
    #probar = 'drive'
    #
    if True:
        
        ''' data loading '''
        if probar == 'drive':
            saved = r'C:\Users\ghior\Documents\MATLAB\DeepApproachHtransformation\Data\MedidasDeep\data512img\yT_'
            toSave = r'C:\Users\ghior\Documents\MATLAB\DeepApproachHtransformation\DeepLearning\10teamH\outDrive'
            samples = 151
            offset = 0
            modely1.save_weights('modelosEntrenados/UNetYtoY1epD'+str(epocas)+'.h5')
        if probar == 'ARAD':
            saved = r'C:\Users\ghior\Documents\MATLAB\DeepApproachHtransformation\Data\MedidasDeep\aradValidationF\yT_'
            toSave = r'C:\Users\ghior\Documents\MATLAB\DeepApproachHtransformation\DeepLearning\10teamH\outArad'
            samples = 10
            offset = 451
            modely1.save_weights('modelosEntrenados/UNetYtoY1epA'+str(epocas)+'.h5')
        
        hy_list = []
        hy1_list = []
        hy2_list = []
        for i in np.arange(samples):
            file = saved + str((i+offset)) + '.mat'
            temp_m = loadmat(file)
     
            inImaBCC_00 = temp_m['re']
            Ref_img=np.expand_dims(inImaBCC_00,0)
            outImaBCC_00 = modely1.predict(Ref_img,batch_size=1)
            
            Resul=outImaBCC_00[0,:,:,:]
                    
            out_path = toSave + '\outa_' + str(i+offset) + '.mat'
            
            scipy.io.savemat(out_path, dict(y_pred=Resul))
            
    return modely1
    
