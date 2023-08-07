
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
#from keras.utils import np_utils

from tensorflow.keras.layers import Input, Flatten, Dense, MaxPooling2D, GlobalAveragePooling2D, Reshape, Conv2D, Dropout

from scatternet.kymatioex.morlet2d import ReducedMorletScattering2D #StarletScattering2D, ShapeletScattering2D
from scatternet.utils.classifier import check_classifier
from scatternet.utils.dataset import RadioGalaxies, Galaxy10, MINST, Mirabest, MirabestBinary
from scatternet.utils.classifier import check_classifier, ClassifierNN
from kymatio.keras import Scattering2D

ScaNet = ReducedMorletScattering2D
d = MirabestBinary() #
d.truncate_train(20,balance = True) 
d.augment()


#================================================

#the optimal 2^J is of the order of the maximum pixel displacements due to translations and deformations. 

J,L = 3,8#3,8
scanet = ScaNet( J,L, max_order = 2, subsample=True)
#scanet = Scattering2D(J, L)
print("Using J = {0} scales, L = {1} angles".format(J,L))


inputs = Input(shape=d.data_shape)
x = inputs
x = scanet(x)
scamodel = Model(inputs, x)
d.preprocess( scamodel.predict )
#d.preprocess( lambda a: np.sum(scamodel.predict(a),axis=(2,3)) )



#x = tf.math.reduce_sum(x,axis=(2,3))
#x = Dense(64,activation = 'relu')(x)
#x = Dense(32,activation = 'relu')(x)

inputs = Input(shape=d.data_shape)
x = inputs
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
model.summary()

clf = ClassifierNN(model, d)
clf.fit(d.x_train, d.y_train)
check_classifier(clf, d.x_test, d.y_test, d.label_list, "Classic Scattering + NN, test data")
#check_classifier(clf, d.x_test_rot, d.y_test, d.label_list, "Classic Scattering + NN, test data rotated 90 deg")





