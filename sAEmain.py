#  Structure Deep Learning project by HDSP group
# Any information contacted with jorge.bacca1@correo.uis.edu.co



#-------------------- This are main py------------------------

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

# To run in cpu
os.environ["CUDA_VISIBLE_DEVICES"]= '-1'
''' data set 2 '''
PATH = r'C:\Users\ghior\Documents\MATLAB\DeepApproachHtransformation\Data\MedidasDeep\aradTrainF'
# parameters of the net
BATCH_SIZE = 2; IMG_WIDTH = 140; IMG_HEIGHT = 128; L_bands    = 8; L_imput    = 8; split_v = 0.2

tra_ep = [64,128,256,512]
#epocas = 100
epocas = 64
test_dataset,train_dataset=Build_data_set(IMG_HEIGHT,IMG_WIDTH,L_bands,L_imput,BATCH_SIZE,PATH,split_v)  # build the DataStore from Read_Spectral.py

#-------------Net_model----------------------------------------------------------------
pretrained_weights1 = r'modelosEntrenados/DenoiseEp64.h5'
#modelae = Denoise(pretrained_weights=pretrained_weights1,input_size=(IMG_HEIGHT,IMG_WIDTH,L_bands))
modelae = Denoise(input_size=(IMG_HEIGHT,IMG_WIDTH,L_bands))

optimizad = tf.keras.optimizers.Adam(learning_rate=1e-4, amsgrad=False)
modelae.compile(optimizer=optimizad, loss='mean_squared_error',run_eagerly=False,metrics=[psnr,'mse','mae',SSIM])

#model.load_weights("UNetAdaptx2.h5")



history = modelae.fit_generator(train_dataset, epochs=epocas,shuffle=True,validation_data=(test_dataset))

#.------------ seee the accuracy---------------
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('aea.png')
plt.savefig('UNetx2Train.eps', dpi=500)
plt.show()

modelae.save_weights('modelosEntrenados/DenoiseEp'+str(epocas)+'.h5')
#model.save_weights("UNetAdaptx4.h5")


test_loss = modelae.evaluate(test_dataset,  verbose=2)


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
        modely1.save_weights('modelosEntrenados/DenoiseEp'+str(epocas)+'.h5')
    if probar == 'ARAD':
        saved = r'C:\Users\ghior\Documents\MATLAB\DeepApproachHtransformation\Data\MedidasDeep\aradValidationF\yT_'
        toSave = r'C:\Users\ghior\Documents\MATLAB\DeepApproachHtransformation\DeepLearning\10teamH\outArad'
        samples = 10
        offset = 451
        modely1.save_weights('modelosEntrenados/DenoiseEp'+str(epocas)+'.h5')
    hy_list = []
    hy1_list = []
    hy2_list = []
    for i in np.arange(samples):
        file = saved + str((i)) + '.mat'
        temp_m = loadmat(file)
 
        inImaBCC_00 = temp_m['re']
        Ref_img=np.expand_dims(inImaBCC_00,0)
        outImaBCC_00 = modelae.predict(Ref_img,batch_size=1)
        
        Resul=outImaBCC_00[0,:,:,:]
                
        out_path = toSave + '\outa_' + str(i) + '.mat'
        
        scipy.io.savemat(out_path, dict(y_pred=Resul))
    
    

