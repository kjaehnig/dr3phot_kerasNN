import pandas as pd
import tensorflow as tf
import numpy as np
import astroquery as aq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_clsts_w_params():
    clsts_w_params = pd.read_csv("/Users/karljaehnig/Repositories/MIST_isochrone_widget/mist_isochrone_params.csv")
    ann_clsts = clsts_w_params.loc[clsts_w_params.feh < 10]    # ignore all clusters with 999 entries
    return ann_clsts


def load_dr3phot():
    dr3phot = pd.read_feather("/Users/karljaehnig/Repositories/MIST_isochrone_widget/cg2020dr3phot.feather")
    dr3phot.loc[:,'bp_rp'] = dr3phot.phot_bp_mean_mag.values - dr3phot.phot_rp_mean_mag.values
    dr3phot = dr3phot.loc[
                        (np.isfinite(dr3phot.bp_rp.values))
                        &
                        (np.isfinite(dr3phot.phot_g_mean_mag.values))
                        ]

    return dr3phot


def generate_cmd_pixel_dfs(bins2d=[10,10]):
    pixdict = {}
    for ii in range(product(bins2d)):
        pixdict[f'p{ii}'] = []
    pixdict['age'] = []
    pixdict['dist'] = []
    pixdict['av'] = []
    pixdict['feh'] = []
    pixdict['cluster'] = []

    for ii,clst in tqdm(enumerate(ann_clsts.cluster.str.lower().values),leave=None,position=2):
        clstphot = dr3phot.loc[dr3phot.cluster.str.lower().values == clst]
        isoparams = ann_clsts.iloc[ii]
        feh = isoparams['feh']
        dist = isoparams['distance']
        av = isoparams['av']
        age = isoparams['age']
        if clstphot.shape[0] > 6:
            res = np.histogram2d(clstphot.bp_rp.values,
                         clstphot.phot_g_mean_mag.values,
                         bins=bins2d,
                         density=False
                         )
            pvals = res[0].T.ravel()
            for ii in tqdm(range(len(pvals)), position=1,leave=None):
                pixdict[f'p{ii}'].append(pvals[ii])
            pixdict['age'].append(age)
            pixdict['dist'].append(dist)
            pixdict['av'].append(av)
            pixdict['feh'].append(feh)
            pixdict['cluster'].append(clst)
    pxdf = pd.DataFrame(pixdict, columns=pixdict.keys())

    return pxdf



def generate_X_Y_arrs(pxdf, bins2d=[10,10]):

    npix = np.product(bins2d)

    data = pxdf.values[:,:-1].astype(np.float64)

    X = data[:,0:npix]

    Y = data[:,npix:]


    return (X,Y)


def generate_scaled_test_train_arrs(X,Y, train_size=0.8, Scaler):

    xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=1-train_size)

    scalerx = Scaler()
    scalery = Scaler()

    xtrain_scale = scalerx.fit_transform(xtrain)
    xtest_scale = scalerx.fit_transform(xtest)

    ytrain_scale = scalery.fit_transform(ytrain)
    ytest_scale = scalery.fit_transform(ytest)

    res = {
        'xtrain':xtrain,
        'xtest':xtest,
        'ytrain':ytrain,
        'ytest':ytest,
        'xtrain_scale':xtrain_scale,
        'xtest_scale':xtest_scale,
        'ytrain_scale':ytrain_scale,
        'ytest_scale':ytest_scale,
        'scalerx':scalerx,
        'scalery':scalery
        }

    return res

