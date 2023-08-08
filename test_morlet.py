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

#d = RadioGalaxies()
d = MINST()
x = d.x_train[d._unique_indices]

#================================================

M,J,L = d.dim_x, 2,4

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

#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axs = plt.subplots(len(d._unique_indices) +1, 1+J*L, sharex=True, sharey=True, figsize = (8,5))
#plt.set_cmap('cubehelix')


axs[0,0].set_axis_off()
for i, filter in enumerate(filters_set['psi']):
    f = filter["levels"][0]
    #filter_c = tf.signal.fft2d(f*x_U, name='fft2d')
    filter_c = tf.signal.fft2d(f, name='fft2d')
    filter_c = np.fft.fftshift(filter_c)
    #filter_c = np.abs(filter_c)
    axs[0,i+1].imshow(colorize(filter_c))
    axs[0,i+1].get_xaxis().set_ticks([])
    axs[0,i+1].get_yaxis().set_ticks([])
    #axs[0,i+1].set_title("j = {0},theta={1}".format(i // L, i % L))
    axs[0,i+1].set_title("$j = {}$ \n $\\theta={}$".format(i // L, i % L))

# fig.suptitle((r"Wavelets for each scales $j$ and angles $\theta$ used."
#               "\nColor saturation and color hue respectively denote complex "
#               "magnitude and complex phase."), fontsize=13)


x_U = tf.signal.fft2d(x, name='fft2d')
plt.set_cmap('cubehelix')

f = filters_set['phi']["levels"][0]

# filter_c = tf.signal.fft2d(f, name='fft2d')
# filter_c = np.fft.fftshift(filter_c)
# filter_c = np.abs(filter_c)

filter_c2 = tf.signal.fft2d(f*x, name='fft2d')
filter_c2 = np.fft.fftshift(filter_c2)
filter_c2 = np.abs(filter_c2)

print(filter_c2.shape)

for i in range(len(d._unique_indices)):
    axs[i+1,0].imshow(x[i,:,:])
    for j, filter in enumerate(filters_set['psi']):
        f = filter["levels"][0]
        filter_c = tf.signal.fft2d(f*x[i,:,:], name='fft2d')
        filter_c = np.fft.fftshift(filter_c)
        filter_c = np.abs(filter_c)
        axs[i+1,j+1].imshow(filter_c)





fig.show()

plt.show()

sys.exit()



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





