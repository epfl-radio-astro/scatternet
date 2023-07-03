
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from keras.utils import np_utils

from tensorflow.keras.layers import Input, Flatten, Dense, MaxPooling2D, AveragePooling2D, Reshape, Conv2D, Dropout

from scatternet.kymatioex.morlet2d import ReducedMorletScattering2D #StarletScattering2D, ShapeletScattering2D
from scatternet.utils.classifier import check_classifier
from scatternet.utils.dataset import RadioGalaxies, Galaxy10, MINST
from scatternet.utils.classifier import check_classifier, ClassifierNN
from kymatio.keras import Scattering2D

ScaNet = ReducedMorletScattering2D
d = RadioGalaxies(add_channel = True) #
d.truncate_train(50) 
d.augment()

print(d.class_weights)
#================================================

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
model.summary()

clf = ClassifierNN(model, d)
clf.fit(d.x_train, d.y_train)

check_classifier(clf, d.x_test, d.y_test, d.label_list, "Classic CNN, test data")
check_classifier(clf, d.x_test_rot, d.y_test, d.label_list, "Classic CNN, test data rotated 90 deg")





