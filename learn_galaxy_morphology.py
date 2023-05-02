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

from tensorflow.keras.models import Model
from keras.utils import np_utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.cluster import KMeans
from sklearn import svm, metrics
from sklearn.utils.fixes import loguniform

from tensorflow.keras.layers import Input, Flatten, Dense, AveragePooling2D, Reshape

from scatternet.kymatioex.ExtendedScattering2D import ReducedMorletScattering2D, StarletScattering2D, ShapeletScattering2D
from scatternet.utils.data_processing import format_galaxies, check_data_processing
from scatternet.utils.plotting import plot_features

ScaNet = ShapeletScattering2D


#================================================
label_list = ['Disk, Face-on, No Spiral', 'Smooth, Completely round', 'Smooth, in-between round', 'Smooth, Cigar shaped', 'Disk, Edge-on, Rounded Bulge', 'Disk, Edge-on, Boxy Bulge', 
            'Disk, Edge-on, No Bulge','Disk, Face-on, Tight Spiral', 'Disk, Face-on, Medium Spiral', 'Disk, Face-on, Loose Spiral']

#images, labels = galaxy10.load_data()

with h5py.File('data/Galaxy10.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])
Y = labels.astype(np.intc)
X = images.astype(np.intc)

keys, unique_indices = np.unique(Y, return_index = True)
(n_train_samples, dim_x, dim_y, __), n_classes = X.shape, keys.size

X = format_galaxies(X, threshold = 0.1, min_size = 100, margin = 2)

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

#check_data_processing(x_train, x_train_clean, y_train, unique_indices, label_list)
#sys.exit()

#================================================

def create_model():
    J,L = 3,8
    scanet = ScaNet( 10,20, max_order = 1)
    print("Using J = {0} scales, L = {1} angles".format(J,L))
    inputs = Input(shape=(dim_x, dim_y))
    #scanet = ScaNet(J=J, L=L)
    x = Reshape([dim_x,dim_y,1])(inputs)
    x = AveragePooling2D((2,2), name = "avgpool")(x)
    x = Reshape([int(dim_x/2),int(dim_y/2)])(x)
    x = scanet(x)
    x = tf.math.reduce_sum(x,axis=(2,3))
    #x = Dense(64, activation='relu')(x)
    x = Dense(len(unique_indices), activation='sigmoid')(x)
    model = Model(inputs, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
 
estimator = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))




