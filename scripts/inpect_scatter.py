
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
#from keras.utils import np_utils

from tensorflow.keras.layers import Input, Flatten, Dense, AveragePooling2D, Reshape

from scatternet.kymatioex.morlet2d import ReducedMorletScattering2D #StarletScattering2D, ShapeletScattering2D
from kymatio.keras import Scattering2D
#from scatternet.utils.data_processing import format_galaxies, check_data_processing
from scatternet.utils.plotting import plot_features
from scatternet.utils.classifier import check_classifier, ClassifierSVC
from scatternet.utils.dataset import RadioGalaxies, Galaxy10, MINST, Mirabest, MirabestBinary

#ScaNet = ReducedMorletScattering2D
d = RadioGalaxies()
d.truncate_train(10, balance = True) 
#d.augment()

#================================================

#the optimal 2^J is of the order of the maximum pixel displacements due to translations and deformations. 

J,L = 3,2#3,8
#scanet = ScaNet( J,L, max_order = 2)
print("Using J = {0} scales, L = {1} angles".format(J,L))

inputs = Input(shape=(d.dim_x, d.dim_y))

x = inputs
x = Scattering2D( J,L, max_order = 2)(x)
model = Model(inputs, x)
model.summary()

print("Now predicting")
feature_matrix = model.predict(d.x_train)
#feature_labels = scanet.labels(inputs)






#========================================================


n_output_coeffs = feature_matrix.shape[1]

s0_labels = ["Order 0: low-pass"]
s1_labels = []
s2_labels = []
for j1 in range(J):
    for l1 in range(L):
        s1_labels.append("Order 1: j = {0}, $\\theta$ = {1}".format(j1,l1))
        for j2 in range(J):
            if j2 <= j1: continue
            for l2 in range(L):
                s2_labels.append("Order 2 j = {0},{1}, $\\theta$ = {2},{3}".format(j1,j2,l1,l2))
feature_labels = s0_labels + s1_labels + s2_labels

print("ScaNet has {0} output coefficients with dimension {1}".format(n_output_coeffs,feature_matrix.shape))
print(feature_labels)

fig, axs = plt.subplots(len(d.x_train),n_output_coeffs+1, figsize=(10, 6))
plt.set_cmap('cubehelix')
plt.tick_params(left = False, bottom=False)
#plt.tight_layout()
fig.subplots_adjust(hspace=0, wspace= 0, bottom = 0.01, left = 0.1, top = 0.7,  right = 0.9)

for idx, gal in enumerate(d.x_train):

    if idx == 0:
        v = axs[idx,0].set_title("Input", fontsize=10)
        v.set_rotation(70)
        for i,f in enumerate(feature_labels):
            print(idx, len(d.x_train), i+1, n_output_coeffs+1)
            vi = axs[idx,i+1].set_title(f, fontsize=10,ha = 'left')
            vi.set_rotation(70)
    axs[idx,0].imshow(gal)
    axs[idx,0].set_yticklabels([])
    axs[idx,0].set_yticks([])
    axs[idx,0].set_xticks([])
    axs[idx,0].set_xticklabels([])
    #h = axs[idx,0].set_ylabel( d.label_list[d.x_train[idx]],fontsize=10,loc = 'top')
    h = axs[idx,0].set_ylabel( d.label_list[d.y_train[idx]],fontsize=10, ha = 'right')
    h.set_rotation(0)    

    for i,f in enumerate(range(n_output_coeffs)):
        axs[idx,i+1].imshow(feature_matrix[idx,f,:,:])
        axs[idx,i+1].set_yticklabels([])
        axs[idx,i+1].set_xticklabels([])
        axs[idx,i+1].set_yticks([])
        axs[idx,i+1].set_xticks([])

plt.show()



