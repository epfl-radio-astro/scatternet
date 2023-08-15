import sys
import os
import json
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import svm, metrics

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Flatten, Dense, MaxPooling2D, GlobalAveragePooling2D, Reshape, Conv2D, Dropout

from scatternet.kymatioex.morlet2d import ReducedMorletScattering2D #StarletScattering2D, ShapeletScattering2D
from scatternet.utils.classifier import check_classifier
from scatternet.utils.dataset import RadioGalaxies, Galaxy10, MINST, Mirabest, MirabestBinary
from scatternet.utils.classifier import check_classifier, ClassifierNN, ClassifierSVC
from kymatio.keras import Scattering2D


def make_cnn_clf(d,outdir):
    inputs = Input(shape=(d.dim_x, d.dim_y,1))

    # strucutre following https://arxiv.org/pdf/1807.10406.pdf

    x = inputs
    x = Conv2D(6,(5,5),activation='relu')(x)
    x = MaxPooling2D( (2,2), strides = (2,2))(x)
    x = Conv2D(16,(5,5),activation='relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Flatten()(x)
    x = Dense(120,activation = 'relu')(x)
    x = Dense(84,activation = 'relu')(x)
    x = Dropout(0.5)(x)
    if len(d.label_list) <= 2:
        x = Dense(1, activation='sigmoid')(x)
    else:
        x = Dense(len(d.label_list) , activation='softmax')(x)

    model = Model(inputs, x)

    clf = ClassifierNN(model, d, outdir)
    return clf

def make_cnn_clf_detailed(d,outdir):
    inputs = Input(shape=(d.dim_x, d.dim_y,1))

    # strucutre following https://arxiv.org/pdf/1807.10406.pdf

    x = inputs
    x = Conv2D(6,(5,5),activation='relu')(x)
    x = MaxPooling2D( (2,2), strides = (2,2))(x)
    x = Conv2D(16,(5,5),activation='relu')(x)
    x = MaxPooling2D( (2,2), strides = (2,2))(x)
    #x = Conv2D(64,(5,5),activation='relu')(x)
    #x = GlobalAveragePooling2D()(x)

    x = Flatten()(x)
    x = Dense(120,activation = 'relu')(x)
    x = Dense(84,activation = 'relu')(x)
    x = Dropout(0.5)(x)
    if len(d.label_list) <= 2:
        x = Dense(1, activation='sigmoid')(x)
    else:
        x = Dense(len(d.label_list) , activation='softmax')(x)

    model = Model(inputs, x)

    clf = ClassifierNN(model, d,outdir)
    return clf

def make_nn_clf(d,outdir):

    inputs = Input(shape=d.data_shape)
    x = inputs
    #x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(120,activation = 'relu')(x)
    x = Dense(84,activation = 'relu')(x)
    x = Dropout(0.5)(x)
    if len(d.label_list) <= 2:
        x = Dense(1, activation='sigmoid')(x)
    else:
        x = Dense(len(d.label_list) , activation='softmax')(x)
    model = Model(inputs, x)

    clf = ClassifierNN(model, d,outdir)
    return clf

def scatter_data(s,d):
    inputs = Input(shape=(d.dim_x, d.dim_y))
    x = inputs
    x = s(x)
    model = Model(inputs, x)
    #model.summary()
    d.preprocess( model.predict )

def eval(clf, d):
    
    X = d.x_test
    y = d.y_test
    y_pred = clf.predict(X)
    acc = metrics.balanced_accuracy_score(y, y_pred)
    f1 = metrics.f1_score(y,y_pred,average = 'weighted')
    return acc, f1

def print_status(tag, acc, f1):
    
    print("############################################")
    print("# {0} accuracy = {1:.2f}, f1 score = {2:.2f}".format(tag,acc,f1))
    print("############################################")

if __name__ == "__main__":
    outname = "results_compare"
    outdir = "./results_compare/{0}/".format(outname)
    try:
       os.makedirs(outdir)
    except FileExistsError:
       # directory already exists
       pass
    print("Writing results to", outdir)

    J,L = 3,8#3,8
    scanet_reduced = ReducedMorletScattering2D( J,L, max_order = 2)
    scanet   = Scattering2D(J, L, max_order = 2)
    wavenet  = Scattering2D(J, L, max_order = 1) 

    

    clf_keys = ['cnn', 'cnn2','svm', 'redscattersvm','scattersvm', 'wavesvm', 'redscatternet','scatternet', 'wavenet']
    classifiers = {}

    d1 = RadioGalaxies()
    d1.write("data/RadioGalaxies.npy")
    sys.exit()
    d1.augment()
    d1.save_original()
    d2 = MirabestBinary()
    d2.save_original()

    # trim datasets to be the same size
    d2.preprocess( lambda x:  x[:,30:90,30:90] )

    results = {}
    for k in clf_keys:
        results[k] = {}

    for i in range(5):
        print("###################################################################")
        print("#Trial {0}".format(i))
        print("###################################################################")

        for k in clf_keys: results[k][i]  = {}
        for d in [d1,d2]:

            # classifier acting on original data
            if d == d1:
                classifiers['svm'] = ClassifierSVC()
                classifiers['svm'].fit(d.x_train,d.y_train)
            acc, f1 = eval(classifiers['svm'], d)
            results['svm'][i]['acc'] = acc
            results['svm'][i]['f1']  = f1
            print_status('svm', acc, f1)

            # pass data through scattering2d        
            scatter_data(scanet,d)

            if d == d1:
                classifiers['scattersvm'] = ClassifierSVC()
                classifiers['scattersvm'].fit(d.x_train,d.y_train)
            acc, f1 = eval(classifiers['scattersvm'], d)
            results['scattersvm'][i]['acc'] = acc
            results['scattersvm'][i]['f1']  = f1
            print_status('scattersvm', acc, f1)

            if d == d1:
                classifiers['scatternet'] = make_nn_clf(d,outdir)
                classifiers['scatternet'].fit(d.x_train,d.y_train,d.x_val, d.y_val)
            acc, f1 = eval(classifiers['scatternet'], d)
            results['scatternet'][i]['acc'] = acc
            results['scatternet'][i]['f1']  = f1
            print_status('scatternet', acc, f1)
            
            # pass data through reduced scattering2d  
            d.restore_original()
            scatter_data(scanet_reduced,d)
            print("test",d.x_train.shape)
            
            if d == d1:
                classifiers['redscattersvm'] = ClassifierSVC()
                classifiers['redscattersvm'].fit(d.x_train,d.y_train)
            acc, f1 = eval(classifiers['redscattersvm'], d)
            results['redscattersvm'][i]['acc'] = acc
            results['redscattersvm'][i]['f1']  = f1

            if d == d1:
                classifiers['redscatternet'] = make_nn_clf(d,outdir)
                classifiers['redscatternet'].fit(d.x_train,d.y_train,d.x_val, d.y_val)
            acc, f1 = eval(classifiers['redscatternet'], d)
            results['redscatternet'][i]['acc'] = acc
            results['redscatternet'][i]['f1']  = f1
            print_status('redscatternet', acc, f1)

            # pass data through 1st order wavelet scattering  
            d.restore_original()
            scatter_data(wavenet,d)
            print("test",d.x_train.shape)

            if d == d1:
                classifiers['wavesvm'] = ClassifierSVC()
                classifiers['wavesvm'].fit(d.x_train,d.y_train)
            acc, f1 = eval(classifiers['wavesvm'], d)
            results['wavesvm'][i]['acc'] = acc
            results['wavesvm'][i]['f1']  = f1

            if d == d1:
                classifiers['wavenet'] = make_nn_clf(d,outdir)
                classifiers['wavenet'].fit(d.x_train,d.y_train,d.x_val, d.y_val)
            acc, f1 = eval(classifiers['wavenet'], d)
            results['wavenet'][i]['acc'] = acc
            results['wavenet'][i]['f1']  = f1
            print_status('wavenet', acc, f1)

            d.restore_original()
            d.add_channel()

            if d == d1:
                classifiers['cnn'] = make_cnn_clf(d,outdir)
                classifiers['cnn'].fit(d.x_train,d.y_train,d.x_val, d.y_val)
            acc, f1 = eval(classifiers['cnn'], d)
            #print("{0} training examples, acc = {1:.2f}, f1 = {2:.2f}".format(n_train, acc, f1))
            results['cnn'][i]['acc'] = acc
            results['cnn'][i]['f1']  = f1
            print_status("cnn",acc,f1)

            if d == d1:
                classifiers['cnn2'] = make_cnn_clf_detailed(d,outdir)
                classifiers['cnn2'].fit(d.x_train,d.y_train,d.x_val, d.y_val)
            acc, f1 = eval(classifiers['cnn2'], d)
            #print("{0} training examples, acc = {1:.2f}, f1 = {2:.2f}".format(n_train, acc, f1))
            results['cnn2'][i]['acc'] = acc
            results['cnn2'][i]['f1']  = f1
            print_status("cnn2",acc,f1)

            with open("{0}/{1}.json".format(outdir,outname), 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)




        