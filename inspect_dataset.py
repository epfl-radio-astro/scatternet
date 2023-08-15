'''
Galaxy morphology classification on the Galaxy10 DECals Dataset:
https://astronn.readthedocs.io/en/latest/galaxy10.html
Follow the instructions at the link above to download Galaxy10.h5
'''
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorsys import hls_to_rgb

from scatternet.utils.dataset import RadioGalaxies, Galaxy10, MINST, Mirabest, MirabestBinary
from scatternet.kymatioex.morlet2d import ReducedMorletScattering2D #StarletScattering2D, ShapeletScattering2D
from kymatio.scattering2d.filter_bank import filter_bank

ScaNet = ReducedMorletScattering2D

def add_noise(img,noise_level=0.1):
        img_shape = img.shape
        img = img.flatten()
        for i,c in enumerate(img):
            img[i] = c + np.random.normal(scale=noise_level)
        return img.reshape(img_shape)

def plot_dataset():
    #================================================

    n_examples = 3
    #d = MirabestBinary()
    #d = MINST()
    d = RadioGalaxies()
    d.info()
    x,y= d.get_example_classes(n_examples)


    #================================================

    plt.rc('font', family='serif')
    fig, axs = plt.subplots(n_examples,len(d._unique_indices), sharex=True, sharey=True, figsize = (8,5))
    plt.set_cmap('cubehelix')

    for n in range(n_examples):
        for i in range(len(d._unique_indices)):
            axs[n,i].imshow(x[i*n_examples+n,:,:])
            if n==0:
                axs[n,i].set_title(d.label_list[i])
            axs[n,i].get_xaxis().set_ticks([])
            axs[n,i].get_yaxis().set_ticks([])





    fig.show()

    plt.show()

def plot_noise_injection():
    noise_levels = [0,0.01, 0.1, 0.5]
    d = RadioGalaxies()
    x,y= d.get_example_classes(1)
    plt.rc('font', family='serif')
    fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize = (8,5))
    axs = axs.flatten()
    plt.set_cmap('cubehelix')

    selected_image = x[3,:,:]

    for i,n in enumerate(noise_levels):
        axs[i].imshow( add_noise(selected_image,n))
        axs[i].set_title('Noise = {0}'.format(n))
        axs[i].get_xaxis().set_ticks([])
        axs[i].get_yaxis().set_ticks([])


    fig.show()
    plt.show()

plot_noise_injection()



