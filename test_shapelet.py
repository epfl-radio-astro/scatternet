import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, AveragePooling2D, Reshape

from scatternet.kymatioex.ExtendedScattering2D import ShapeletScattering2D
from scatternet.utils.data_processing import format_galaxies, check_data_processing

def plot_shapelets(s):

    print('test',s.shape)
    n, nx, ny = s.shape

    n_rt = int(n**0.5)
    if n != n_rt*n_rt:
        n_rt += 1

    fig, axs = plt.subplots(n_rt,n_rt, figsize=(10, 6))
    axs = axs.flatten()
    plt.tick_params(left = False, bottom=False)
    #plt.tight_layout()
    #fig.subplots_adjust(hspace=0, wspace= 0, bottom = 0.01, left = 0.25, top = 0.7,  right = 0.99)


    for i in range(n):
        axs[i].imshow(s[i,:,:])



    #n_output_coeffs = feature_matrix.shape[1]


    # for idx, gal in enumerate(indices):

    #     if idx == 0:
    #         v = axs[idx,0].set_title("Original input image", fontsize=10)
    #         v1 = axs[idx,1].set_title("Cleaned input image", fontsize=10)
    #         v.set_rotation(70)
    #         v1.set_rotation(70)
    #         for i,f in enumerate(feature_labels):
    #             vi = axs[idx,i+2].set_title(f, fontsize=10)
    #             vi.set_rotation(70)
    #     axs[idx,0].imshow(x0[gal,:,:])
    #     axs[idx,0].set_yticklabels([])
    #     axs[idx,0].set_xticklabels([])
    #     h = axs[idx,0].set_ylabel( label_list[y_in[gal]],fontsize=10,loc = 'top')
    #     h.set_rotation(-10)

    #     axs[idx,1].imshow(x_in[gal,:,:])
    #     axs[idx,1].set_yticklabels([])
    #     axs[idx,1].set_xticklabels([])
        

    #     for i,f in enumerate(range(n_output_coeffs)):
    #         axs[idx,i+2].imshow(feature_matrix[gal,f,:,:])
    #         axs[idx,i+2].set_yticklabels([])
    #         axs[idx,i+2].set_xticklabels([])

    plt.show()


label_list = ['Disk, Face-on, No Spiral', 'Smooth, Completely round', 'Smooth, in-between round', 'Smooth, Cigar shaped', 'Disk, Edge-on, Rounded Bulge', 'Disk, Edge-on, Boxy Bulge', 
            'Disk, Edge-on, No Bulge','Disk, Face-on, Tight Spiral', 'Disk, Face-on, Medium Spiral', 'Disk, Face-on, Loose Spiral']

#images, labels = galaxy10.load_data()

with h5py.File('data/Galaxy10.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])
labels = labels.astype(np.intc)
images = images.astype(np.intc)

keys, unique_indices = np.unique(labels, return_index = True)
(n_train_samples, dim_x, dim_y, __), n_classes = images.shape, keys.size

images_clean = format_galaxies(images, threshold = 0.1, min_size = 100, margin = 2)

# parameters are beta and n
scatternet = ShapeletScattering2D( 10,20)

inputs = Input(shape=(dim_x, dim_y))

#scanet = ScaNet(J=J, L=L)

x = scatternet(inputs)
shapelet_filters = scatternet.filters
print(shapelet_filters.shape)

plot_shapelets(shapelet_filters)
print(shapelet_filters.shape, images_clean.shape )

print(np.max(shapelet_filters), np.max(images_clean))


#plot_shapelets(shapelet_filters[:,:,:]*images_clean[None, 0,:,:])
#plot_shapelets(shapelet_filters[:,:,:]*images_clean[None, 1,:,:])

coeffs = np.sum(shapelet_filters*images_clean[None, 0,:,:], axis = (1,2))
print(coeffs.shape)
images_reco = np.sum(coeffs[:,None,None]*shapelet_filters,axis = 0)

fig, axs = plt.subplots(2,1, figsize=(10, 6))
axs = axs.flatten()
axs[0].imshow(images_clean[0,:,:])
axs[1].imshow(images_reco)
plt.show()


sys.exit()

model = Model(inputs, x)
model.summary()

print("Now predicting")
feature_matrix = model.predict(images_clean[0:10,:,:])
#feature_labels = scanet.labels(inputs)
feature_labels = scatternet.labels()
