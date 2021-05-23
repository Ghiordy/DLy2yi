import tensorflow as tf  # se puede cambiar por from keras.import backend as K
from tensorflow.keras.layers import Layer  # quitar tensorflow si usa keras solo
from tensorflow.keras.constraints import NonNeg
import numpy as np
import poppy
import os
from random import random
from scipy.io import loadmat
from functions import deta, ifftshift, area_downsampling_tf, compl_exp_tf, transp_fft2d, transp_ifft2d, img_psf_conv,fftshift2d_tf,get_color_bases,propagation,propagation_back,kronecker_product
from tensorflow.keras.constraints import NonNeg


class CASSI_Layer(Layer):

    def __init__(self, output_dim, M=256, N = 256, L = 12,Nt=8,wave_lengths=None,**kwargs):

        self.output_dim = output_dim
        self.M = M
        self.N = N
        self.L = L
        self.Nt = Nt

        if wave_lengths is not None:
            self.wave_lengths = wave_lengths
        else:
            self.wave_lengths = np.linspace(420, 660, 12)*1e-9

        self.fr, self.fg, self.fc, self.fb = get_color_bases(self.wave_lengths)

        super(CASSI_Layer, self).__init__(**kwargs)

    def build(self, input_shape):

        wr = (np.random.rand(self.Nt, self.Nt))
        wg = (np.random.rand(self.Nt, self.Nt))
        wb = (np.random.rand(self.Nt, self.Nt))
        wc = (np.random.rand(self.Nt, self.Nt))
        wt = wr + wg + wb + wc
        wr =  tf.constant_initializer(tf.math.divide(wr, wt))
        wg =  tf.constant_initializer(tf.math.divide(wg, wt))
        wb =  tf.constant_initializer(tf.math.divide(wb, wt))
        wc =  tf.constant_initializer(tf.math.divide(wc, wt))

        self.wr = self.add_weight(name='wr', shape=(self.Nt,self.Nt, 1),
                                  initializer=wr, trainable=True, constraint=NonNeg())
        self.wg = self.add_weight(name='wg', shape=(self.Nt,self.Nt, 1),
                                  initializer=wg, trainable=True, constraint=NonNeg())
        self.wb = self.add_weight(name='wb', shape=(self.Nt,self.Nt, 1),
                                  initializer=wb, trainable=True, constraint=NonNeg())
        self.wc = self.add_weight(name='wc', shape=(self.Nt, self.Nt, 1),
                                  initializer=wc, trainable=True, constraint=NonNeg())

        super(CASSI_Layer, self).build(input_shape)

    def call(self, inputs, **kwargs):

        wt = self.wr + self.wg + self.wb + self.wc
        wr = tf.math.divide(self.wr, wt)
        wg = tf.math.divide(self.wg, wt)
        wb = tf.math.divide(self.wb, wt)
        wc = tf.math.divide(self.wc, wt)

        Aux1 = tf.multiply(wr, self.fr) + tf.multiply(wg, self.fg) + tf.multiply(wb, self.fb) + tf.multiply(wc, self.fc)
        Mask = kronecker_product(
            tf.ones(int(self.M/self.Nt),int(self.M/self.Nt)), Aux1)

        #         Mask 1x256x256x12
        #          Input 16x256x256x12
        #Y_aux = []

        #for i in range(self.L)
            #Aux = tf.math.mulply(Mask,Input)
            #Y_aux = tf.concat()



        y_med_r = tf.reduce_sum(y_med_r, axis=3)
        y_med_r = tf.expand_dims(y_med_r, -1)
        y_med_g = tf.reduce_sum(y_med_g, axis=3)
        y_med_g = tf.expand_dims(y_med_g, -1)
        y_med_b = tf.reduce_sum(y_med_b, axis=3)
        y_med_b = tf.expand_dims(y_med_b, -1)

        y_final = tf.concat([y_med_r, y_med_g, y_med_b], axis=3)


        return y_final

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)