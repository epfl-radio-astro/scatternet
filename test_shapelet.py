import h5py
import numpy as np
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet
from scipy import ndimage
import matplotlib.pyplot as plt
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util


class ShapeletBasis(ShapeletSet):
    def __init__(self): 
        ShapeletSet.__init__(self)   

    def getbasis(self, x, y, n_max, beta, deltaPix, center_x=0, center_y=0):
        """
        decomposes an image into the shapelet coefficients in same order as for the function call
        :param x:
        :param y:
        :param n_max:
        :param beta:
        :param center_x:
        :param center_y:
        :return:
        """
        num_param = int((n_max+1)*(n_max+2)/2)
        print(len(x), len(y))
        base_list = np.zeros( (num_param, len(x)))
        amp_norm = 1./beta**2*deltaPix**2
        n1 = 0
        n2 = 0
        H_x, H_y = self.shapelets.pre_calc(x, y, beta, n_max, center_x, center_y)
        for i in range(num_param):
            kwargs_source_shapelet = {'center_x': center_x, 'center_y': center_y, 'n1': n1, 'n2': n2, 'beta': beta, 'amp': amp_norm}
            base = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
            base_list[i,:] = base
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        return base_list



def shapeletanalysis(img):
    # we slightly convolve the image with a Gaussian convolution kernel of a few pixels (optional)
    #sigma = 5
    print(img.shape[0]*img.shape[0],'input pixels')
    #ngc_conv = ndimage.filters.gaussian_filter(img, sigma, mode='nearest', truncate=6)
    ngc_conv = img

    # we now degrate the pixel resoluton by a factor.
    # This reduces the data volume and increases the spead of the Shapelet decomposition
    factor = 1  # lower resolution of image with a given factor
    numPix_large = int(len(ngc_conv)/factor)
    n_new = int((numPix_large-1)*factor)
    ngc_cut = ngc_conv[0:n_new,0:n_new]

    print('npix', numPix_large-1)
    x, y = util.make_grid(numPix=numPix_large-1, deltapix=1)  # make a coordinate grid
    ngc_data_resized = image_util.re_size(ngc_cut, factor)  # re-size image to lower resolution


    print(x,y)
    # now we come to the Shapelet decomposition
    # we turn the image in a single 1d array
    image_1d = util.image2array(ngc_data_resized)  # map 2d image in 1d data array

    #image_1d = util.image2array()  # map 2d image in 1d data array

    # we define the shapelet basis set we want the image to decompose in
    n_max = 10  # choice of number of shapelet basis functions, 150 is a high resolution number, but takes long
    beta = 10  # shapelet scale parameter (in units of resized pixels)

    shapeletSet = ShapeletBasis()
    basis_ngc = shapeletSet.getbasis(x, y, n_max, beta, 1., center_x=0, center_y=0) 

    print( [b.shape for b in basis_ngc])

    coeff_ngc = [np.sum(b*image_1d) for b in basis_ngc]
    print(image_1d.shape)
    print( [b.shape for b in coeff_ngc])
    print(len(coeff_ngc), 'number of coefficients')  # number of coefficients

    print(coeff_ngc)

    # reconstruct NGC1300 with the shapelet coefficients
    image_reconstructed = shapeletSet.function(x, y, coeff_ngc, n_max, beta, center_x=0, center_y=0)
    # turn 1d array back into 2d image
    image_reconstructed_2d = util.array2image(image_reconstructed)  # map 1d data vector in 2d image

    f, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=False, sharey=False)

    ax = axes[0]
    im = ax.matshow(img, origin='lower')
    ax.set_title("original image")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)

    ax = axes[1]
    im = ax.matshow(ngc_conv, origin='lower')
    ax.set_title("convolved image")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)

    ax = axes[2]
    im = ax.matshow(ngc_data_resized, origin='lower')
    ax.set_title("resized")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)

    ax = axes[3]
    im = ax.matshow(image_reconstructed_2d, origin='lower')
    ax.set_title("reconstructed")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)

    plt.show()


with h5py.File('data/Galaxy10.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])
labels = labels.astype(np.intc)
images = images.astype(np.intc)
images = np.sum(images / 255./3., axis = -1)

shapeletanalysis(images[0])
shapeletanalysis(images[8])

