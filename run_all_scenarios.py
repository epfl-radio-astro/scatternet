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
    clf.fit(d.x_train,d.y_train,d.x_val,d.y_val, verbose = 0)
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
    name = "mbb"
    do_augment = False
    set_n_train = [20,40,60,100,200,300, 400, 500] 
    outname = "results_mirabest_binary"

    print(sys.argv)
    if len(sys.argv) > 3:
        print("Program does not understand extra arguments. Expected input:\npython pipeline_full.py {mb, mbb, galaxy, minst} {augment: 0 or 1 (default 0)} ")
        sys.exit()
    elif len(sys.argv) <= 1:
        print("No command line arguments received. Running Mirabest binary data with no augmentation.")
    else:
        name = sys.argv[1]
        if len(sys.argv) > 2:
            do_augment = True
    print(do_augment)

    
    n_trial = 5
    if name == 'mbb':
        set_n_train = [20,40,60,100,200,300, 400, 500] 
        outname = "results_mirabestbinary"
    elif name == 'mb':
        set_n_train = [10,20,50,100,150,200,300] 
        outname = "results_mirabest"
    elif name == 'galaxy':
        set_n_train = [20,40,60,100,200,300, 400, 500, 750, 1000] 
        outname = "results_galaxy"
    elif name == 'minst':
        set_n_train = [20,40,100,200,500,1000, 1500, 2000, 3000, 4000,6000, 10000]
        outname = "results_minst"

    if do_augment: 
        outname = outname.replace("results_","results_aug_")


    outdir = "./results/{0}/".format(outname)
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

    clf_svc = ClassifierSVC()

    clf_keys = ['cnn', 'cnn2','svm', 'redscattersvm','scattersvm', 'wavesvm', 'redscatternet','scatternet', 'wavenet']
    #results = {}
    #for k in clf_keys: results[k] = {}

    for n_train in set_n_train:
        print("\n########################################################################################")
        print("########################################################################################")
        print("## {0} Training samples".format(n_train))
        print("########################################################################################")
        print("########################################################################################\n")

        results = {}
        for k in clf_keys: results[k] = {}

        for i in range(n_trial):
            for k in clf_keys: results[k][i]  = {}

            print("###################################################################")
            print("#Trial {0} with {1} training data points".format(i,n_train))
            print("###################################################################")

            # set up dataset
            #d = RadioGalaxies() #
            if name == 'galaxy':
                d = RadioGalaxies()
            elif name == 'minst':
                d = MINST() #
            elif name == 'mb':
                d = Mirabest()
            elif name == 'mbb':
                d = MirabestBinary()
            
            d.truncate_train(n_train, balance = True, randomize=True) 
            if do_augment: d.augment()
            d.save_original()

            # classifier acting on original data
            acc, f1 = eval(clf_svc, d)
            results['svm'][i]['acc'] = acc
            results['svm'][i]['f1']  = f1
            print_status('svm', acc, f1)

            # pass data through scattering2d        
            scatter_data(scanet,d)
            acc, f1 = eval(clf_svc, d)
            results['scattersvm'][i]['acc'] = acc
            results['scattersvm'][i]['f1']  = f1
            print_status('scattersvm', acc, f1)

            acc, f1 = eval(make_nn_clf(d,outdir), d)
            results['scatternet'][i]['acc'] = acc
            results['scatternet'][i]['f1']  = f1
            print_status('scatternet', acc, f1)
            
            # pass data through reduced scattering2d  
            d.restore_original()
            scatter_data(scanet_reduced,d)
            print("test",d.x_train.shape)
            acc, f1 = eval(clf_svc, d)
            results['redscattersvm'][i]['acc'] = acc
            results['redscattersvm'][i]['f1']  = f1

            acc, f1 = eval(make_nn_clf(d,outdir), d)
            results['redscatternet'][i]['acc'] = acc
            results['redscatternet'][i]['f1']  = f1
            print_status('redscatternet', acc, f1)

            # pass data through 1st order wavelet scattering  
            d.restore_original()
            scatter_data(wavenet,d)
            print("test",d.x_train.shape)
            acc, f1 = eval(clf_svc, d)
            results['wavesvm'][i]['acc'] = acc
            results['wavesvm'][i]['f1']  = f1

            acc, f1 = eval(make_nn_clf(d,outdir), d)
            results['wavenet'][i]['acc'] = acc
            results['wavenet'][i]['f1']  = f1
            print_status('wavenet', acc, f1)

            d.restore_original()

            d.add_channel()
            cnn_clf = make_cnn_clf(d,outdir)
            acc, f1 = eval(cnn_clf, d)
            print("{0} training examples, acc = {1:.2f}, f1 = {2:.2f}".format(n_train, acc, f1))
            results['cnn'][i]['acc'] = acc
            results['cnn'][i]['f1']  = f1
            print_status("cnn",acc,f1)

            cnn_clf2 = make_cnn_clf_detailed(d,outdir)
            acc, f1 = eval(cnn_clf2, d)
            print("{0} training examples, acc = {1:.2f}, f1 = {2:.2f}".format(n_train, acc, f1))
            results['cnn2'][i]['acc'] = acc
            results['cnn2'][i]['f1']  = f1
            print_status("cnn2",acc,f1)

            with open("{0}/{1}_{2}.json".format(outdir,outname, n_train), 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)




        