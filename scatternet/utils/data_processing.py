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
        #x_clean[i,:,:] = remove_artifacts(x[i,:,:], threshold, min_size, margin)
        x_clean[i,:,:] = flood_select(x[i,:,:], threshold, min_size, margin)
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


def flood_select(im, threshold = 0.3, min_size = 100, margin = 5, normalize = True):
    if normalize == True:
        im = im/np.max(im)
    binary_img = np.full(im.shape, False)
    visited = np.zeros(im.shape)
    center_x, center_y = [int(s/2) for s in im.shape]
    
    #binary_img[center_x, center_y] = 1

    import sys
    sys.setrecursionlimit(2500)

    def _flood_fill(x ,y, prev, prev_min, lim = 1000):

        if lim == 0: return
        lim -= 1

        if x < 0 or x >= im.shape[0] or y < 0 or y >= im.shape[1]:
            return

        if visited[x,y]: return
        visited[x,y] = True
        binary_img[x,y] = 1

        if im[x,y] > threshold:

            #if prev < im[x,y]*.9-0.05: return
            if im[x,y]*.9 > prev_min: return
            #if im[x,y]*.95 - 0.05 > prev_min: return

            
            prev_min = min(prev, im[x,y])
            prev = im[x,y]


            #print(x,y,binary_img[x,y], im[x,y] > threshold, visited[x,y] )

            for r in [x-1, x, x+1]:
                for c in [y-1, y, y+1]:
                    if x == r and y == c:
                        continue
                    _flood_fill(r, c, prev, prev_min, lim)

            #_flood_fill(x+1, y, prev, prev_min, lim)
            #_flood_fill(x-1, y, prev, prev_min, lim)
            #_flood_fill(x, y+1, prev, prev_min, lim)
            #_flood_fill(x, y-1, prev, prev_min, lim)

        return


    _flood_fill(center_x,center_y, 1., 1.)
    '''label_im, number_of_objects = ndimage.label(im > threshold,np.ones((3,3)))
    sizes = ndimage.sum(im, label_im, range(number_of_objects + 1))
    mask = sizes > min_size
    binary_img = mask[label_im]
    binary_img = morphology.binary_erosion(binary_img,morphology.disk(margin, dtype=bool))'''
    return binary_img*im

def check_data_processing(x_original, x_clean, y, unique_indices, label_list):
    fig, axs = plt.subplots(2,len(unique_indices), figsize=(10, 6))
    fig.subplots_adjust(hspace=0, wspace= 0, bottom = 0.01, left = 0.1, top = 0.7,  right = 0.99)

    for idx, gal in enumerate(unique_indices):

        axs[0,idx].imshow(x_original[gal,:,:])
        axs[0,idx].set_yticklabels([])
        axs[0,idx].set_xticklabels([])
        h = axs[0,idx].set_title( label_list[y[gal]],fontsize=10,loc = 'center')
        h.set_rotation(-10)

        axs[1,idx].imshow(x_clean[gal,:,:])
        axs[1,idx].set_yticklabels([])
        axs[1,idx].set_xticklabels([])

    plt.show()

