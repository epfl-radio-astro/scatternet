
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

ScaNet = ReducedMorletScattering2D
d = MirabestBinary(add_channel = True) #
#d.remove_uncertain_classes()
d.truncate_train(20, balance = True) 
d.augment()

print(d.class_weights, d.x_train.shape, d.y_train.shape)
#================================================

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

clf = ClassifierNN(model, d)
clf.fit(d.x_train, d.y_train)

check_classifier(clf, d.x_test, d.y_test, d.label_list, "Classic CNN, test data")
check_classifier(clf, d.x_train, d.y_train, d.label_list, "Classic CNN, test data")
#check_classifier(clf, d.x_test_rot, d.y_test, d.label_list, "Classic CNN, test data rotated 90 deg")





