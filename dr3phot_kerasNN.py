import pandas as pd
import numpy as np
import maplotlib.pyplot as plt

import tensorflow as tf
import helper_functions as hf

from sklearn.preprocessing import StandardScaler

bins2d = [30,30]

ann_clsts = hf.load_clsts_w_params()

dr3phot = hf.load_dr3phot()

pxdf = hf.generate_cmd_pixel_dfs(bins2d=bins2d)

X,Y = hf.generate_X_Y_arrs(pxdf, bins2d)

res = hf.generate_scaled_test_train_arrs(X,Y,StandardScaler)

xtrain_scale = res['xtrain_scale']
ytrain_scale = res['ytrain_scale']

def base_model(X,Y):
    ninputs = X.shape[1]
    noutputs= Y.shape[1]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
                int(ninputs)+1, input_shape=(int(ninputs),),
                kernel_initializer='normal',
                activation='relu'
                ))
    model.add(tf.keras.layers.Dense(
                500,
                activation='relu'
                ))
    model.add(tf.keras.layers.Dense(
                250,
                activation='relu'
                ))
    model.add(tf.keras.layers.Dense(
                100,
                activation='relu'
                ))
    model.add(tf.keras.layers.Dense(
                50,
                activation='relu'
                ))
    model.add(tf.keras.layers.Dense(
                noutputs,
                activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


mdl = base_model(X,Y)

mdl.fit(xtrain_scale, ytrain_scale, 
    epochs=150, 
    batch_size=25, 
    callbacks=tf.keras.callbacks.EarlyStopping(
        monitor='loss', 
        patience=10, 
        start_from_epoch=50)
)

