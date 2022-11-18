
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
#import tensorflow_transform as tft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.cluster import KMeans
from sklearn import svm, metrics
from sklearn.utils.fixes import loguniform

from tensorflow.keras.layers import Input, Flatten, Dense

#from astroNN.datasets import galaxy10
#from astroNN.datasets.galaxy10 import galaxy10cls_lookup

from TestScattering2D import ReducedScattering2D, StarletScattering2D
from data_processing import format_galaxies 
from plotting import plot_features

# None = no scattering

ScaNet = StarletScattering2D


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

keys, unique_indices = np.unique(y_train, return_index = True)
(n_train_samples, dim_x, dim_y, __), n_classes = x_train.shape, keys.size

x_train_clean = format_galaxies(x_train, threshold = 0.3, min_size = 100, margin = 2)
x_test_clean  = format_galaxies(x_test, threshold = 0.3, min_size = 100, margin = 2)


#================================================

if ScaNet == ReducedScattering2D:
    J,L = 3,8
    inputs = Input(shape=(dim_x, dim_y))
    print("Using J = {0} scales, L = {1} angles".format(J,L))
    scanet = ScaNet(J=J, L=L)
    x     = scanet(inputs)
    model = Model(inputs, x)
    model.compile()
    print("Now predicting")
    feature_matrix = model.predict(x_train_clean)
    feature_labels = scanet.labels(inputs)

else:
    J = 3
    print("Using J = {0} scales".format(J))
    scanet = ScaNet(J=J)
    feature_matrix = scanet.predict(x_train_clean)
    feature_labels = scanet.labels()





n_output_coeffs = feature_matrix.shape[1]
print("ScaNet has {0} output coefficients with dimension {1}x{2}".format(n_output_coeffs,feature_matrix.shape[2],feature_matrix.shape[3]))

#========================================================
'''plt.clf()
plt.tick_params(left = False, bottom=False)
fig, axs = plt.subplots(len(unique_indices),n_output_coeffs+2, figsize=(10, 6))
#plt.tight_layout()
fig.subplots_adjust(hspace=0, wspace= 0, bottom = 0.01, left = 0.25, top = 0.7,  right = 0.99)

for idx, gal in enumerate(unique_indices):

    if idx == 0:
        v = axs[idx,0].set_title("Original input image", fontsize=10)
        v1 = axs[idx,1].set_title("Cleaned input image", fontsize=10)
        v.set_rotation(70)
        v1.set_rotation(70)
        for i,f in enumerate(feature_labels):
            vi = axs[idx,i+2].set_title(f, fontsize=10)
            vi.set_rotation(70)
    axs[idx,0].imshow(x_train[gal,:,:])
    axs[idx,0].set_yticklabels([])
    axs[idx,0].set_xticklabels([])
    h = axs[idx,0].set_ylabel( label_list[y_train[gal]],fontsize=10,loc = 'top')
    h.set_rotation(-10)

    axs[idx,1].imshow(x_train_clean[gal,:,:])
    axs[idx,1].set_yticklabels([])
    axs[idx,1].set_xticklabels([])
    

    for i,f in enumerate(range(n_output_coeffs)):
        axs[idx,i+2].imshow(feature_matrix[gal,f,:,:])
        axs[idx,i+2].set_yticklabels([])
        axs[idx,i+2].set_xticklabels([])

plt.show()'''
plot_features(x_train, x_train_clean, y_train, feature_matrix, unique_indices, feature_labels, label_list)
#plot_features(x_train, x_train_clean, y_train, feature_matrix, np.where(y_train==0), feature_labels, label_list)
sys.exit()

n_features = feature_matrix.shape[1]*feature_matrix.shape[2]*feature_matrix.shape[3]
n_components = min(n_features/2., 150)
print("doing {0}-component pca on {1} features".format(n_components, n_features))
pca = PCA(n_components)
X_scaled = StandardScaler().fit_transform(feature_matrix.reshape(feature_matrix.shape[0],-1))
X_pca = pca.fit_transform(X_scaled)
X_pca = X_pca[:,:50]

param_grid = {
    "C": loguniform(1e3, 1e5),
    #"gamma": loguniform(1e-4, 1e-1),
}

print("Fitting SVC")
#kernel = rbf
#clf = RandomizedSearchCV(
#    svm.SVC(kernel="linear", class_weight="balanced"), param_grid, n_iter=10
#)
clf = svm.SVC(kernel="linear", class_weight="balanced")
clf.fit(X_pca, y_train)


def check_classifier(clf, X, y, label_list):

    #t0 = time()
    y_pred = clf.predict(X)
    #print("done in %0.3fs" % (time() - t0))

    print(metrics.classification_report(y, y_pred, target_names=label_list))
    metrics.ConfusionMatrixDisplay.from_estimator(
        clf, X, y, display_labels=label_list, xticks_rotation="vertical"
    )
    plt.tight_layout()
    plt.show()

check_classifier(clf, X_pca, y_train, label_list)

'''print("Now double-checking prediction")
feature_matrix0 = feature_matrix[0:500,:,:]
X0 = StandardScaler().fit_transform(feature_matrix0.reshape(feature_matrix0.shape[0],-1))
X0_pca = pca.transform(X0)
check_classifier(clf, X0_pca, y_train[0:500], label_list)'''

print("Now predicting test data")
feature_matrix_test = model.predict(x_test_clean)
X_test_scaled = StandardScaler().fit_transform(feature_matrix_test.reshape(feature_matrix_test.shape[0],-1))
X_test_pca = pca.transform(X_test_scaled)
X_test_pca = X_test_pca[:,:50]
check_classifier(clf, X_test_pca, y_test,label_list)




