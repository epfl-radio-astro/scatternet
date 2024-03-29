import numpy as np
import tensorflow as tf
import h5py
import sys
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import class_weight

import random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

def shuffle(x,y):
    temp = list(zip(x, y))
    random.shuffle(temp)
    x,y = zip(*temp)
    x = np.array(x)
    y = np.array(y)
    return x,y

class DataSet():
    '''
    Abstract class that defines the functionality of DataSets in this library
    '''
    def __init__(self, add_channel = False):
   
        self.modified = False
        self.load_data()
        self.keys, self._unique_indices = np.unique(self.y_train, return_index = True)
        if add_channel:
            self.add_channel()
        else:
            (self._n_train, self._dim_x, self._dim_y), self.n_classes = self.x_train.shape, self.keys.size
        self._x_test_rot = np.rot90(self.x_test,axes=(1, 2))
        self._encoder = LabelBinarizer()
        self._encoder.fit(self.y_train)


    def write(self,outname):
        with open(outname, 'wb') as f:
            np.save(f, self.x_train)
            np.save(f, self.y_train)
            np.save(f, self.x_val)
            np.save(f, self.y_val)
            np.save(f, self.x_test)
            np.save(f, self.y_test)
            np.save(f,self.label_list)

    def save_original(self):
        self.__x_train    = np.copy(self._x_train)
        self.__x_test     = np.copy(self._x_test)
        self.__x_test_rot = np.copy(self._x_test_rot)
        self.__x_val      = np.copy(self._x_val)

    def restore_original(self):
        self._x_train    = self.__x_train
        self._x_test     = self.__x_test
        self._x_test_rot = self.__x_test_rot
        self._x_val      = self.__x_val

    def add_channel(self):
        self._x_train = self._x_train.reshape(*self._x_train.shape,1)
        self._x_test  = self._x_test.reshape( *self._x_test.shape,1)
        self._x_val   = self._x_val.reshape(  *self._x_val.shape,1)
    
        (self._n_train, self._dim_x, self._dim_y,__), self.n_classes = self.x_train.shape, self.keys.size

    def load_data(self):
        raise NotImplementedError

    def preprocess(self,f):
        self._x_train    = f(self._x_train)
        self._x_test     = f(self._x_test)
        self._x_test_rot = f(self._x_test_rot)
        self._x_val      = f(self._x_val)

    def info(self):

        for i, k in enumerate(self.keys):
            n_train = len(np.where(self.y_train == i)[0])
            n_test = len(np.where(self.y_test == i)[0])
            n_val = len(np.where(self.y_val == i)[0])
            print(self.label_list[i], "n_train =", n_train, "n_val =", n_val, "n_test =", n_test)

    def truncate_train(self,n, balance = False, randomize = False):

        if randomize:
            self._x_train, self._y_train = shuffle(self._x_train, self._y_train)

        if balance:
            n_per_class = int(n / self.n_classes)
            indices = []
            for i, k in enumerate(self.keys):
                class_indices = np.where(self.y_train == i)[0][:n_per_class]
                indices = np.append(indices,class_indices)
            indices = indices.astype(np.intc)
            self._x_train = self._x_train[indices]
            self._y_train = self._y_train[indices]

        else:
            self._x_train = self._x_train[:n]
            self._y_train = self._y_train[:n]


    def get_example_classes(self,n=1):
        indices = []
        for i, k in enumerate(self.keys):
            class_indices = np.where(self.y_train == i)[0][:n]
            indices = np.append(indices,class_indices)
        indices = indices.astype(np.intc)
        return (self._x_train[indices], self._y_train[indices])

    def augment(self):

        print('Augment',self._x_train.shape)

        x_train_flip = np.flip(self.x_train,axis=1)
        x_train = np.vstack( [self._x_train, x_train_flip])
        y_train = np.append(self._y_train,self._y_train)

        for k in range(1,4):
            x_train_rotated         = np.rot90(self._x_train,axes=(1, 2),k=k)
            x_train_rotated_flipped = np.rot90(x_train_flip,axes=(1, 2),k=k)

            x_train = np.vstack( [x_train, x_train_rotated])
            x_train = np.vstack( [x_train, x_train_rotated_flipped])
            y_train = np.append(y_train,self._y_train)
            y_train = np.append(y_train,self._y_train)


        self._x_train, self._y_train = shuffle(x_train, y_train)

        print(self._x_train.shape)


    @property
    def data_shape(self):
        return self._x_test_rot.shape[1:]

    def encode(self, y):
        return self._encoder.transform(y)

    def decode(self, y):
        return self._encoder.inverse_transform(y)

    @property
    def class_weights(self):
        return class_weight.compute_class_weight(class_weight='balanced',
                                                 classes = np.unique(self.y_train),
                                                 y=self.y_train)

    @property
    def sample_weights(self):
        return class_weight.compute_sample_weight('balanced',self.y_train)

    @property
    def label_list(self):
        raise NotImplementedError

    @property
    def unique_indices(self):
        return self._unique_indices

    @property
    def dim_x(self):
        return self._dim_x

    @property
    def dim_y(self):
        return self._dim_y

    @property
    def x_train(self):
        return self._x_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def n_train(self):
        return self._n_train

    @property
    def x_val(self):
        return self._x_val

    @property
    def y_val(self):
        return self._y_val

    @property
    def x_test(self):
        return self._x_test

    @property
    def x_test_rot(self):
        return self._x_test_rot

    @property
    def y_test(self):
        return self._y_test

class DataFromFile(DataSet):
    def __init__(self, infilename, add_channel = False):
        self.infilename = infilename
    def load_data(self):
        with open(self.infilename, 'rb') as f:
            self.x_train = np.load(f)
            self.y_train = np.load(f)
            self.x_val   = np.load(f)
            self.y_val   = np.load(f)
            self.x_test  = np.load(f)
            self.y_test  = np.load(f)
            self._label_list= np.load(f)
    @property
    def label_list(self):
        return self._label_list

class RadioGalaxies(DataSet):
    '''
    This Radio Galaxy Dataset is a collection and combination of several catalogues
    using the FIRST radio galaxy survey [[1]](https://ui.adsabs.harvard.edu/abs/1995ApJ...450..559B/abstract).
    https://www.sciencedirect.com/science/article/pii/S2352340923000926
    '''

    def load_set(self,F, key, x,y):
        data_entry = F[key + "/Img"]
        label_entry = F[key + "/Label_literature"]
        d = np.array(data_entry)
        y.append(np.array(label_entry))
        x.append(d)

    def format(self,x,y, trimby = 120):
        
        x,y = shuffle(x,y)
        x = x.astype(np.intc)
        y = y.astype(np.intc)
        x = x[:,trimby:-trimby,trimby:-trimby]/255.

        return x,y

    def load_data(self):
        self._x_train, self._y_train, self._x_val,  self._y_val, self._x_test,  self._y_test  = [], [], [], [], [], []
        with h5py.File('data/galaxy_data_h5.h5', 'r') as F:
            for key in F.keys():
                split_entry = F[key + "/Split_literature"].asstr()[()]

                if split_entry == "train":
                    self.load_set(F,key,self._x_train, self._y_train)
                elif split_entry == "test":
                    self.load_set(F,key,self._x_test, self._y_test)
                elif split_entry == "valid":
                    self.load_set(F,key,self._x_val, self._y_val)

        self._x_train, self._y_train = self.format(self._x_train, self._y_train)
        self._x_test,  self._y_test  = self.format(self._x_test,  self._y_test)
        self._x_val,   self._y_val   = self.format(self._x_val,   self._y_val)


    @property
    def label_list(self):
        return ["FRI", "FRII", "Compact", "Bent"]

class Galaxy10(DataSet):
    '''
    Galaxy morphology classification on the Galaxy10 DECals Dataset:
    https://astronn.readthedocs.io/en/latest/galaxy10.html
    Follow the instructions at the link above to download Galaxy10.h5
    '''

    def load_data(self):

        with h5py.File('data/Galaxy10.h5', 'r') as F:
            images = np.array(F['images'])
            labels = np.array(F['ans'])
        labels = labels.astype(np.intc)
        images = images.astype(np.intc)

        split1 = 1000#int(len(labels)*9/10.)
        split2 = split1+200#-1
        split3 = split2+200

        self._x_train, self._y_train = images[0:split1,:,:], labels[0:split1]
        self._x_test,  self._y_test = images[split1:split2,:,:], labels[split1:split2]
        self._x_val,  self._y_val = images[split2:split3,:,:], labels[split2:split3]

        self._x_train = self._format_galaxies(self._x_train, threshold = 0.1, min_size = 100, margin = 2)
        self._x_test  = self._format_galaxie(self._x_test,  threshold = 0.1, min_size = 100, margin = 2)
        self._x_val   = self._format_galaxie(self._x_val,   threshold = 0.1, min_size = 100, margin = 2)

    def _format_galaxies(self,x, threshold = 0.3, min_size = 100, margin = 5):
        
        from scipy import ndimage
        from skimage import morphology

        # collapse color channels
        x = x / 255./3.
        x = np.sum(x, axis = -1)

        # remove artifacts
        x_clean = np.zeros(x.shape)
        for i in range(x.shape[0]):
            #x_clean[i,:,:] = self._remove_artifacts(x[i,:,:], threshold, min_size, margin)
            x_clean[i,:,:] = self._flood_select(x[i,:,:], threshold, min_size, margin)
        return x_clean

    def _remove_artifacts(self,im, threshold = 0.3, min_size = 100, margin = 5, normalize = True):
        if normalize == True:
            im = im/np.max(im)
        label_im, number_of_objects = ndimage.label(im > threshold,np.ones((3,3)))
        sizes = ndimage.sum(im, label_im, range(number_of_objects + 1))
        mask = sizes > min_size
        binary_img = mask[label_im]
        binary_img = morphology.binary_erosion(binary_img,morphology.disk(margin, dtype=bool))
        return binary_img*im


    def _flood_select(self,im, threshold = 0.3, min_size = 100, margin = 5, normalize = True):
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
        return binary_img*im

    @property
    def label_list(self):
        return ['Disk, Face-on, No Spiral', 'Smooth, Completely round', 'Smooth, in-between round', 'Smooth, Cigar shaped', 'Disk, Edge-on, Rounded Bulge', 'Disk, Edge-on, Boxy Bulge', 
                'Disk, Edge-on, No Bulge','Disk, Face-on, Tight Spiral', 'Disk, Face-on, Medium Spiral', 'Disk, Face-on, Loose Spiral']

class MINST(DataSet):
    '''
    Classic MINST letters dataset
    '''

    def load_data(self):

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255., x_test / 255.

        self._x_train = x_train[0:10000]
        self._y_train = y_train[0:10000]

        self._x_test = x_test[0:1000]
        self._y_test = y_test[0:1000]

        self._x_val = x_test[1000:2000]
        self._y_val = y_test[1000:2000]


    @property
    def label_list(self):
        return [0,1,2,3,4,5,6,7,8,9]
    
class Mirabest(DataSet):
    '''
    The MiraBest dataset:https://arxiv.org/abs/2305.11108
    '''
    
    def load_data(self):


        filepath = './data/Mirabest/batches/'
        
        with open(filepath + 'batches.meta', 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data['label_names']
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)} 
        self.idx_to_class = {i: _class for i, _class in enumerate(self.classes)}   
        self._x_test,  self._y_test = self.load_file(filepath,['test_batch'])
        self._x_val,   self._y_val = self.load_file(filepath,['data_batch_1'])
        self._x_train, self._y_train = self.load_file(filepath,["data_batch_{0}".format(i) for i in range(2,8)])
      
    def load_file(self,file_dir, file_list):
        data = []
        targets = []
        for file_path in file_list:
            with open(file_dir+ file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])
        data = np.vstack(data).reshape(-1, 150, 150)
        targets = np.array(targets)
        #data = data.transpose((0, 2, 3, 1))
        return data,targets
    
    def remove_uncertain_classes(self):
        uncertain_classes = [3,4,7,9]
        certain_classes = [0, 1,2,5,6,8]
        
        def remove_entries(exclude_list, x, y):
            targets = np.array(y)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            #h1 = np.array(h1_list).reshape(1, -1)
            #h2 = np.array(h2_list).reshape(1, -1)
            #h1_mask = (targets.reshape(-1, 1) == h1).any(axis=1)
            #h2_mask = (targets.reshape(-1, 1) == h2).any(axis=1)
            #targets[h1_mask] = 0 # set all FRI to Class~0
            #targets[h2_mask] = 1 # set all FRII to Class~1
            x = x[exclude_mask]
            y = targets[exclude_mask]
            return x,y
        
        self._x_test, self._y_test  = remove_entries(uncertain_classes,self._x_test, self._y_test)
        self._x_train,self._y_train = remove_entries(uncertain_classes,self._x_train,self._y_train)
        self._x_val,  self._y_val   = remove_entries(uncertain_classes,self._x_val,self._y_val)
        
        self.keys, self._unique_indices = np.unique(self.y_train, return_index = True)
        self.n_classes = self.keys.size
        self._encoder.fit(self.y_train)

    @property
    def label_list(self):
        return [self.idx_to_class[i] for i in self.keys]
    
class MirabestBinary(Mirabest):
    def load_data(self):
        super().load_data()
        
        def merge(x, y):
            targets = np.array(y)
            fr1_classes = [0,1,2]
            fr2_classes=[5,6]
            exclude_list=[3,4,7,8,9]
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            h1 = np.array(fr1_classes).reshape(1, -1)
            h2 = np.array(fr2_classes).reshape(1, -1)
            h1_mask = (y.reshape(-1, 1) == h1).any(axis=1)
            h2_mask = (y.reshape(-1, 1) == h2).any(axis=1)
            
            targets[h1_mask] = 0 # set all FRI to Class~0
            targets[h2_mask] = 1 # set all FRII to Class~1
            x = x[exclude_mask]
            y = targets[exclude_mask]
            return x,y
        self._x_test, self._y_test  = merge(self._x_test, self._y_test)
        self._x_train,self._y_train = merge(self._x_train,self._y_train)
        self._x_val,  self._y_val   = merge(self._x_val,self._y_val)
        
    @property
    def label_list(self):
        return ["FRI","FRII"]