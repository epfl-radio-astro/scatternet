
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from keras.utils import np_utils

from tensorflow.keras.layers import Input, Flatten, Dense, AveragePooling2D, Reshape

from scatternet.kymatioex.morlet2d import ReducedMorletScattering2D #StarletScattering2D, ShapeletScattering2D
from scatternet.utils.data_processing import format_galaxies, check_data_processing
from scatternet.utils.plotting import plot_features
from scatternet.utils.classifier import check_classifier, ClassifierSVC
from scatternet.utils.dataset import RadioGalaxies, Galaxy10, MINST

ScaNet = ReducedMorletScattering2D
d = RadioGalaxies()
d.truncate_train(20) 
d.augment()

#================================================

#the optimal 2^J is of the order of the maximum pixel displacements due to translations and deformations. 

J,L = 3,8#3,8
scanet = ScaNet( J,L, max_order = 2, subsample=True)
print("Using J = {0} scales, L = {1} angles".format(J,L))

inputs = Input(shape=(d.dim_x, d.dim_y))

x = inputs
#x = Reshape([dim_x,dim_y,1])(x)
#x = AveragePooling2D((2,2), name = "avgpool")(x)
#x = Reshape([int(dim_x/2),int(dim_y/2)])(x)
x = scanet(x)
#x = tf.math.reduce_sum(x,axis=(2,3))
model = Model(inputs, x)
model.summary()

print("Now predicting")
feature_matrix = model.predict(d.x_train)
feature_labels = scanet.labels(inputs)

n_output_coeffs = feature_matrix.shape[1]

print("ScaNet has {0} output coefficients with dimension {1}".format(n_output_coeffs,feature_matrix.shape))

#========================================================

#plot_features(d.x_train, d.x_train, d.y_train, feature_matrix, np.append(d.unique_indices), feature_labels, d.label_list)
#for i in range(4):
#    plot_features(d.x_train, d.x_train, d.y_train, feature_matrix, np.where(d.y_train == 0)[:10], feature_labels, d.label_list)



feature_matrix = np.sum(feature_matrix,axis=(2,3))

clf = ClassifierSVC()

'''print("=== Now checking control case, no scattering ===")
clf.fit(x_train, y_train)
check_classifier(clf, x_train, y_train, label_list, "Image input, train")
print("=== Now checking control case, no scattering, test data ===")
check_classifier(clf, x_test, y_test, label_list, "Image input, test")
print("=== Now checking classificaition with rotation, test data ===")
check_classifier(clf, x_test_rot, y_test, label_list,"Rotated image input, test")'''


print("=== Now checking classificaition with scattering ===")
clf.fit(feature_matrix, d.y_train)
check_classifier(clf, feature_matrix, d.y_train, d.label_list, "Scattering input, train")

print("=== Now checking classificaition with scattering  ===")
feature_matrix_test = model.predict(d.x_test)
feature_matrix_test = np.sum(feature_matrix_test,axis=(2,3))
check_classifier(clf, feature_matrix_test, d.y_test, d.label_list, "Scattering input, test")
#print("=== Now checking classificaition with scattering and rotation ===")
#feature_matrix_test_rot = model.predict(d.x_test_rot)
#feature_matrix_test_rot = np.sum(feature_matrix_test_rot,axis=(2,3))
#check_classifier(clf, feature_matrix_test_rot, d.y_test, d.label_list, "Scattering on rotated image input, test")






