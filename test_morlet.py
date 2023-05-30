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


from scatternet.kymatioex.morlet2d import ReducedMorletScattering2D #StarletScattering2D, ShapeletScattering2D


from kymatio.scattering2d.filter_bank import filter_bank

ScaNet = ReducedMorletScattering2D


#================================================
label_list = ['Disk, Face-on, No Spiral', 'Smooth, Completely round', 'Smooth, in-between round', 'Smooth, Cigar shaped', 'Disk, Edge-on, Rounded Bulge', 'Disk, Edge-on, Boxy Bulge', 
            'Disk, Edge-on, No Bulge','Disk, Face-on, Tight Spiral', 'Disk, Face-on, Medium Spiral', 'Disk, Face-on, Loose Spiral']

#images, labels = galaxy10.load_data()

with h5py.File('data/Galaxy10.h5', 'r') as F:
    
    images = np.array(F['images'])
    labels = np.array(F['ans'])
labels = labels.astype(np.intc)
images = images.astype(np.intc)

split1 = 800#int(len(labels)*9/10.)
split2 = split1+800#-1

x_train, y_train = images[0:split1,:,:], labels[0:split1]
x_test,  y_test = images[split1:split2,:,:], labels[split1:split2]

x_train = np.sum(x_train, axis = -1)/ 255./3.

keys, unique_indices = np.unique(y_train, return_index = True)
(n_train_samples, dim_x, dim_y), n_classes = x_train.shape, keys.size

print(keys, unique_indices)

#x_train_clean = format_galaxies(x_train, threshold = 0.1, min_size = 100, margin = 2)
#x_test_clean  = format_galaxies(x_test, threshold = 0.1, min_size = 100, margin = 2)

#check_data_processing(x_train, x_train_clean, y_train, unique_indices, label_list)
#sys.exit()

#================================================


M,J,L = dim_x, 3,8

filters_set = filter_bank(M, M, J, L=L)

def colorize(z):
    n, m = z.shape
    c = np.zeros((n, m, 3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0/(1.0 + abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a, b in zip(A, B)]
    return c

fig, axs = plt.subplots(J, L, sharex=True, sharey=True, figsize = (8,5))
#fig.set_figheight(6)
#fig.set_figwidth(6)
plt.set_cmap('cubehelix')
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

x = x_train[15]
x_U = tf.signal.fft2d(x, name='fft2d')

i = 0
for filter in filters_set['psi']:
    f = filter["levels"][0]
    #filter_c = tf.signal.fft2d(f*x_U, name='fft2d')
    filter_c = tf.signal.fft2d(f, name='fft2d')
    filter_c = np.fft.fftshift(filter_c)
    filter_c = np.abs(filter_c)
    axs[i // L, i % L].imshow(filter_c)
    axs[i // L, i % L].get_xaxis().set_ticks([])
    axs[i // L, i % L].get_yaxis().set_ticks([])
    axs[i // L, i % L].set_title(
        "$j = {}$ \n $\\theta={}$".format(i // L, i % L))
    i = i+1

# fig.suptitle((r"Wavelets for each scales $j$ and angles $\theta$ used."
#               "\nColor saturation and color hue respectively denote complex "
#               "magnitude and complex phase."), fontsize=13)
fig.show()

plt.show()


fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.axis('off')
plt.set_cmap('cubehelix')

f = filters_set['phi']["levels"][0]

filter_c = tf.signal.fft2d(f, name='fft2d')
filter_c = np.fft.fftshift(filter_c)
filter_c = np.abs(filter_c)

filter_c2 = tf.signal.fft2d(f*x, name='fft2d')
filter_c2 = np.fft.fftshift(filter_c2)
filter_c2 = np.abs(filter_c2)
# axs[0].suptitle(("The corresponding low-pass filter, also known as scaling "
#               "function.\nColor saturation and color hue respectively denote "
#               "complex magnitude and complex phase"), fontsize=13)

for a in axs:
    a.get_xaxis().set_ticks([])
    a.get_yaxis().set_ticks([])

axs[0].imshow(x)
axs[1].imshow(filter_c)
axs[2].imshow(filter_c2)

plt.show()





