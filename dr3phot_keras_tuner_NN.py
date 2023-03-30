import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import utils.helper_functions as hf
import keras_tuner

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


    
def base_model(hp):

    # ninputs = X.shape[1]
    # noutputs= Y.shape[1]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
                int(900)+1, input_shape=(900,),
                kernel_initializer='normal',
                activation=hp.Choice('activation',['relu','tanh']),
                ))

    for ii in range(hp.Int("num_layers", 2,10)):
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(f"units_{ii}", min_value=100, max_value=500, step=100),
                activation=hp.Choice('activation',['relu','tanh']),
                )
            )

    model.add(tf.keras.layers.Dense(
                    4,
                    activation=hp.Choice('activation',['relu','tanh','linear'])))
    learning_rate = hp.Choice('lr', [1e-5,1e-4,1e-3,1e-2,1e-1])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mean_squared_error'],
        )

    # model.add(tf.keras.layers.Dense(
        # hp.Int('units',min_value=100, max_value=500, step=100)
        # ))
    return model


def main():
    bins2d = [30,30]

    ann_clsts = hf.load_clsts_w_params()

    dr3phot = hf.load_dr3phot()

    try:
        print("Checking if previously created px dataset exits.")
        pxdf = pd.read_csv(f"/Users/karljaehnig/Repositories/dr3phot_kerasNN/hist2d_{bins2d[0]}_{bins2d[1]}_pixel_df.csv")
        print("px dataset found.")
    except:
        print("px dataset NOT found, generating froms scratch.")
        pxdf = hf.generate_cmd_pixel_dfs(ann_clsts, dr3phot, bins2d=bins2d)

    X,Y = hf.generate_X_Y_arrs(pxdf, bins2d)

    res = hf.generate_scaled_test_train_arrs(X,Y,train_size=0.8,scaler=MinMaxScaler)

    xtrain_scale = res['xtrain_scale']
    ytrain_scale = res['ytrain_scale']

    # mdl = base_model(keras_tuner.HyperParameters())

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=10)

    tuner = keras_tuner.Hyperband(
        hypermodel = base_model,
        objective='mean_squared_error',
        max_epochs=100,
        overwrite=True,
        factor=3,
        directory='/Users/karljaehnig/Repositories/dr3phot_kerasNN/',
        project_name='keras_tuner_search_results'
    )
    tuner.search_space_summary()


    tuner.search(
        xtrain_scale,
        ytrain_scale,
        epochs=50,
        batch_size=50,
        callbacks=[stop_early]
    )

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=5)[0]

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(xtrain_scale, ytrain_scale, epochs=150, callbacks=[stop_early])

    loss_per_epoch = history.history['mean_squared_error']
    best_epoch = loss_per_epoch.index(min(loss_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    mdl = tuner.hypermodel.build(best_hps)

    # Retrain the model
    mdl.fit(xtrain_scale, ytrain_scale, epochs=best_epoch)
    # print(f"""
    # The hyperparameter search is complete. The optimal number of units in the first densely-connected
    # layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    # is {best_hps.get('learning_rate')}.
    # """)
        # mdl.fit(xtrain_scale, ytrain_scale, 
    #     epochs=150, 
    #     batch_size=50, 
    #     callbacks=tf.keras.callbacks.EarlyStopping(
    #         monitor='loss', 
    #         patience=10, 
    #         start_from_epoch=50)
    # )

    hf.predict_and_plot_testvals(res,mdl)
    return (mdl, res)



if __name__ == "__main__":
    main()

