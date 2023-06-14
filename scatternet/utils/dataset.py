import numpy as np
import h5py
from scatternet.utils.data_processing import format_galaxies, check_data_processing

class DataSet():
    def __init__(self):

        self.load_data()
        keys, self._unique_indices = np.unique(self.y_train, return_index = True)
        (self._n_train, self._dim_x, self._dim_y), n_classes = self.x_train.shape, keys.size
        self._x_test_rot = np.rot90(self.x_test,axes=(1, 2))

    def load_data(self):
        raise NotImplementedError

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

class RadioGalaxies(DataSet):
    '''
    This Radio Galaxy Dataset is a collection and combination of several catalogues
    using the FIRST radio galaxy survey [[1]](https://ui.adsabs.harvard.edu/abs/1995ApJ...450..559B/abstract).
    '''

    def load_set(self,F, key, x,y):
        data_entry = F[key + "/Img"]
        label_entry = F[key + "/Label_literature"]
        d = np.array(data_entry)
        y.append(np.array(label_entry))
        x.append(d)

    def format(self,x,y, trimby = 120):
        import random
        temp = list(zip(x, y))
        random.shuffle(temp)
        x, y = zip(*temp)
        x = np.array(x)
        y = np.array(y)
        x = x.astype(np.intc)
        y = y.astype(np.intc)
        x = x[:,trimby:-trimby,trimby:-trimby]

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

        self._x_train = format_galaxies(self._x_train, threshold = 0.1, min_size = 100, margin = 2)
        self._x_test  = format_galaxies(self._x_test,  threshold = 0.1, min_size = 100, margin = 2)
        self._x_val   = format_galaxies(self._x_val,   threshold = 0.1, min_size = 100, margin = 2)


    @property
    def label_list(self):
        return ['Disk, Face-on, No Spiral', 'Smooth, Completely round', 'Smooth, in-between round', 'Smooth, Cigar shaped', 'Disk, Edge-on, Rounded Bulge', 'Disk, Edge-on, Boxy Bulge', 
                'Disk, Edge-on, No Bulge','Disk, Face-on, Tight Spiral', 'Disk, Face-on, Medium Spiral', 'Disk, Face-on, Loose Spiral']

class MINST(DataSet):
    '''
    Galaxy morphology classification on the Galaxy10 DECals Dataset:
    https://astronn.readthedocs.io/en/latest/galaxy10.html
    Follow the instructions at the link above to download Galaxy10.h5
    '''

    def load_data(self):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255., x_test / 255.

    self._x_train = x_train[0:1000]
    self._y_train = y_train[0:1000]

    self._x_test = x_test[0:1000]
    self._y_test = y_test[0:1000]


    @property
    def label_list(self):
        return [0,1,2,3,4,5,6,7,8,9]