import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import svm, metrics

from tensorflow.keras.models import Model
from keras.utils import np_utils

from tensorflow.keras.layers import Input, Flatten, Dense, MaxPooling2D, AveragePooling2D, Reshape, Conv2D, Dropout

from scatternet.kymatioex.morlet2d import ReducedMorletScattering2D #StarletScattering2D, ShapeletScattering2D
from scatternet.utils.classifier import check_classifier
from scatternet.utils.dataset import RadioGalaxies, Galaxy10, MINST
from scatternet.utils.classifier import check_classifier, ClassifierNN, ClassifierSVC
from kymatio.keras import Scattering2D

def make_cnn_clf(d):
    inputs = Input(shape=(d.dim_x, d.dim_y,1))

    # strucutre following https://arxiv.org/pdf/1807.10406.pdf

    x = inputs
    x = Conv2D(64,(6,6),activation='relu')(x)
    x = MaxPooling2D( (2,2), strides = (2,2))(x)
    x = Conv2D(64,(3,3),activation='relu')(x)
    #x = MaxPooling2D( (3,3))(x)
    x = Conv2D(128,(3,3),activation='relu')(x)
    x = Conv2D(256,(3,3),activation='relu')(x)
    x = AveragePooling2D( (4,4))(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(64,activation = 'relu')(x)
    x = Dense(len(d.label_list), activation='softmax')(x)

    model = Model(inputs, x)

    clf = ClassifierNN(model, d)
    return clf

def make_nn_clf(d):

    inputs = Input(shape=d.data_shape)
    x = inputs
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(64,activation = 'relu')(x)
    x = Dense(len(d.label_list), activation='softmax')(x)
    model = Model(inputs, x)

    clf = ClassifierNN(model, d)
    return clf

def scatter_data(s,d):
    inputs = Input(shape=(d.dim_x, d.dim_y))
    x = inputs
    x = s(x)
    model = Model(inputs, x)
    #model.summary()
    d.preprocess( model.predict )

def eval(clf, d):
    clf.fit(d.x_train,d.y_train)
    X = d.x_test
    y = d.y_test
    y_pred = clf.predict(X)
    acc = metrics.balanced_accuracy_score(y, y_pred)
    f1 = metrics.f1_score(y,y_pred,average = 'weighted')
    return acc, f1

n_trial = 10
set_n_train = [20,40,100,200,500,1000, 1500, 2000, 3000, 4000,6000, 10000] 

J,L = 3,8#3,8
scanet_reduced = ReducedMorletScattering2D( J,L, max_order = 2)
scanet   = Scattering2D(J, L, max_order = 2)
wavenet  = Scattering2D(J, L, max_order = 1)

clf_svc = ClassifierSVC()

clf_keys = ['cnn', 'svm', 'redscattersvm','scattersvm', 'wavesvm', 'redscatternet','scatternet', 'wavenet']
results = {}
for k in clf_keys: results[k] = {}

for n_train in set_n_train:
    print("############################################\n# {0} Training samples".format(n_train))
    for k in clf_keys: results[k][n_train] = {}
    for i in range(n_trial):
        for k in clf_keys: results[k][n_train][i]  = {}

        # set up dataset
        #d = RadioGalaxies() #
        d = MINST() #
        d.truncate_train(n_train, balance = True, randomize=True) 
        #d.augment()
        d.save_original()

        # classifier acting on original data
        acc, f1 = eval(clf_svc, d)
        results['svm'][n_train][i]['acc'] = acc
        results['svm'][n_train][i]['f1']  = f1
        print('svm', acc, f1)

        # pass data through scattering2d        
        scatter_data(scanet,d)
        acc, f1 = eval(clf_svc, d)
        results['scattersvm'][n_train][i]['acc'] = acc
        results['scattersvm'][n_train][i]['f1']  = f1
        print('scattersvm', acc, f1)

        acc, f1 = eval(make_nn_clf(d), d)
        results['scatternet'][n_train][i]['acc'] = acc
        results['scatternet'][n_train][i]['f1']  = f1
        print('scatternet', acc, f1)
        
        # pass data through reduced scattering2d  
        d.restore_original()
        scatter_data(scanet_reduced,d)
        print("test",d.x_train.shape)
        acc, f1 = eval(clf_svc, d)
        results['redscattersvm'][n_train][i]['acc'] = acc
        results['redscattersvm'][n_train][i]['f1']  = f1

        acc, f1 = eval(make_nn_clf(d), d)
        results['redscatternet'][n_train][i]['acc'] = acc
        results['redscatternet'][n_train][i]['f1']  = f1
        print('redscatternet', acc, f1)

        # pass data through 1st order wavelet scattering 
        d.restore_original()
        scatter_data(wavenet,d)
        print("test",d.x_train.shape)
        acc, f1 = eval(clf_svc, d)
        results['wavesvm'][n_train][i]['acc'] = acc
        results['wavesvm'][n_train][i]['f1']  = f1

        acc, f1 = eval(make_nn_clf(d), d)
        results['wavenet'][n_train][i]['acc'] = acc
        results['wavenet'][n_train][i]['f1']  = f1
        print('wavenet', acc, f1)

        d.restore_original()

        d.add_channel()
        cnn_clf = make_cnn_clf(d)
        acc, f1 = eval(cnn_clf, d)
        print("{0} training examples, acc = {1:.2f}, f1 = {2:.2f}".format(n_train, acc, f1))
        results['cnn'][n_train][i]['acc'] = acc
        results['cnn'][n_train][i]['f1']  = f1

import json
with open('results_minst.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

        