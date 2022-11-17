import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology

def format_galaxies(x, threshold = 0.3, min_size = 100, margin = 5):
    # collapse color channels
    x = x / 255./3.
    x = np.sum(x, axis = -1)

    # remove artifacts
    x_clean = np.zeros(x.shape)
    for i in range(x.shape[0]):
        x_clean[i,:,:] = remove_artifacts(x[i,:,:], threshold, min_size, margin)
    return x_clean

def remove_artifacts(im, threshold = 0.3, min_size = 100, margin = 5, normalize = True):
    if normalize == True:
        im = im/np.max(im)
    label_im, number_of_objects = ndimage.label(im > threshold,np.ones((3,3)))
    sizes = ndimage.sum(im, label_im, range(number_of_objects + 1))
    mask = sizes > min_size
    binary_img = mask[label_im]
    binary_img = morphology.binary_erosion(binary_img,morphology.disk(margin, dtype=bool))
    return binary_img*im

def check_data_processing(x_original, x_clean, unique_indices, label_list):
    fig, axs = plt.subplots(2,len(unique_indices), figsize=(10, 6))
    fig.subplots_adjust(hspace=0, wspace= 0, bottom = 0.01, left = 0.1, top = 0.7,  right = 0.99)

    for idx, gal in enumerate(unique_indices):

        axs[0,idx].imshow(x_original[gal,:,:])
        axs[0,idx].set_yticklabels([])
        axs[0,idx].set_xticklabels([])
        h = axs[0,idx].set_title( label_list[y_train[gal]],fontsize=10,loc = 'center')
        h.set_rotation(-10)

        axs[1,idx].imshow(x_clean[gal,:,:])
        axs[1,idx].set_yticklabels([])
        axs[1,idx].set_xticklabels([])

    plt.show()

