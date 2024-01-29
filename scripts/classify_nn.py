
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Flatten, Dense, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,Reshape, Conv2D, Dropout

from scatternet.kymatioex.morlet2d import ReducedMorletScattering2D #StarletScattering2D, ShapeletScattering2D
from scatternet.utils.classifier import check_classifier
from scatternet.utils.dataset import RadioGalaxies, Galaxy10, MINST, Mirabest, MirabestBinary
from scatternet.utils.classifier import check_classifier, ClassifierNN
from kymatio.keras import Scattering2D
import visualkeras
from collections import defaultdict

color_map = defaultdict(dict)
color_map[Input]['fill'] = 'lime'
color_map[Conv2D]['fill'] = 'mediumturquoise'
#color_map[ZeroPadding2D]['fill'] = 'gray'
color_map[Dropout]['fill'] = 'gray'
color_map[MaxPooling2D]['fill'] = 'tomato'
color_map[Dense]['fill'] = 'dodgerblue'
color_map[Flatten]['fill'] = 'royalblue'
color_map[GlobalAveragePooling2D]['fill'] = 'darkorange'

ScaNet = ReducedMorletScattering2D
d = RadioGalaxies(add_channel = True) #
#d.remove_uncertain_classes()
#d.truncate_train(20, balance = True) 
d.augment()

print(d.class_weights, d.x_train.shape, d.y_train.shape)
#================================================

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
    model.summary()
    #visualkeras.layered_view(model,one_dim_orientation='y', legend=True, color_map=color_map).show() # display using your system viewer

    clf = ClassifierNN(model, d, outdir,)
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
    model.summary()
    #visualkeras.layered_view(model,one_dim_orientation='y', max_xy = 700, legend=True, color_map=color_map).show() # display using your system viewer

    clf = ClassifierNN(model, d,outdir)
    return clf

clf = make_cnn_clf_detailed(d,'.')
#sys.exit()
clf.fit(d.x_train, d.y_train,d.x_val, d.y_val,1)

check_classifier(clf, d.x_train, d.y_train, d.label_list, "Classic CNN, train data")
check_classifier(clf, d.x_test,  d.y_test, d.label_list, "Large CNN2")

#check_classifier(clf, d.x_test_rot, d.y_test, d.label_list, "Classic CNN, test data rotated 90 deg")





