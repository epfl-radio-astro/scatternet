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

sys.exit()





