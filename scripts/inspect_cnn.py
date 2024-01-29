
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Flatten, Dense, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,Reshape, Conv2D, Dropout

from scatternet.utils.classifier import check_classifier, ClassifierNN
from scatternet.utils.dataset import RadioGalaxies, Galaxy10, MINST, Mirabest, MirabestBinary

def make_cnn_clf_detailed(d):
    inputs = Input(shape=(d.dim_x, d.dim_y,1))

    # strucutre following https://arxiv.org/pdf/1807.10406.pdf

    x = inputs
    x = Conv2D(6,(5,5),activation='relu')(x)
    x = MaxPooling2D( (2,2), strides = (2,2))(x)
    x = Conv2D(16,(5,5),activation='relu')(x)
    x = MaxPooling2D( (2,2), strides = (2,2))(x)
    #x = Conv2D(64,(5,5),activation='relu')(x)
    #x = GlobalAveragePooling2D()(x)

    x = Flatten()(x)
    x = Dense(120,activation = 'relu')(x)
    x = Dense(84,activation = 'relu')(x)
    x = Dropout(0.5)(x)
    if len(d.label_list) <= 2:
        x = Dense(1, activation='sigmoid')(x)
    else:
        x = Dense(len(d.label_list) , activation='softmax')(x)

    model = Model(inputs, x)
    model.summary()
    #visualkeras.layered_view(model,one_dim_orientation='y', max_xy = 700, legend=True, color_map=color_map).show() # display using your system viewer

    clf = ClassifierNN(model, d, outname = "detailed_model_checkpoint")
    return clf

def normalize(f):
    f_min, f_max = f.min(), f.max()
    return (f - f_min) / (f_max - f_min)

d = RadioGalaxies(add_channel=True)
clf = make_cnn_clf_detailed(d)

try:
    clf.model.load_weights("detailed_model_checkpoint")

except:
    print("Training CNN")
    d.augment()
    clf.fit(d.x_train, d.y_train,d.x_val, d.y_val,1)


#check_classifier(clf, d.x_train, d.y_train, d.label_list, "Classic CNN, train data")
#check_classifier(clf, d.x_test,  d.y_test, d.label_list, "Large CNN2")

# summarize feature map shapes
for i in range(len(clf.model.layers)):
    layer = clf.model.layers[i]
    # check for convolutional layer
    if 'conv' not in layer.name: continue
    # summarize output shape
    print(i, layer.name, layer.output.shape)


model1 = Model(inputs=clf.model.inputs, outputs=clf.model.layers[2].output)
model2 = Model(inputs=clf.model.inputs, outputs=clf.model.layers[4].output)
n_img = 5
img = d.x_test[0:n_img,:,:,:]
#img = img.reshape(1,*img.shape)

feature_maps = normalize(model1.predict(img))
feature_maps2 = normalize(model2.predict(img))
n_features1 = feature_maps.shape[-1]
n_features2 = feature_maps2.shape[-1]

filters, biases   = clf.model.layers[1].get_weights()
filters2, biases2 = clf.model.layers[3].get_weights()
# normalize filter values to 0-1 so we can visualize them
filters = normalize(filters)
filters2 = normalize(filters2)
print(filters.shape, filters2.shape)


fig, axs = plt.subplots(n_img,n_features1 + n_features2 + 1, figsize=(10, 6))
fig.subplots_adjust(hspace=0, wspace= 0)
plt.set_cmap('cubehelix')
for i in range(n_img):
    axs[i,0].imshow(img[i,:,:,0])
    axs[i,0].set_yticklabels([])
    axs[i,0].set_xticklabels([])
    axs[i,0].set_yticks([])
    axs[i,0].set_xticks([])
    for f in range(n_features1 + n_features2):
        if f < n_features1:
            axs[i,f+1].imshow(feature_maps[i,:,:,f])
        else: 
            axs[i,f+1].imshow(feature_maps2[i,:,:,f-n_features1])
        axs[i,f+1].set_yticklabels([])
        axs[i,f+1].set_xticklabels([])
        axs[i,f+1].set_yticks([])
        axs[i,f+1].set_xticks([])
plt.show()
sys.exit()







#========================================================


n_output_coeffs = feature_matrix.shape[1]

s0_labels = ["Order 0: low-pass"]
s1_labels = []
s2_labels = []
for j1 in range(J):
    for l1 in range(L):
        s1_labels.append("Order 1: j = {0}, $\\theta$ = {1}".format(j1,l1))
        for j2 in range(J):
            if j2 <= j1: continue
            for l2 in range(L):
                s2_labels.append("Order 2 j = {0},{1}, $\\theta$ = {2},{3}".format(j1,j2,l1,l2))
feature_labels = s0_labels + s1_labels + s2_labels

print("ScaNet has {0} output coefficients with dimension {1}".format(n_output_coeffs,feature_matrix.shape))
print(feature_labels)

fig, axs = plt.subplots(len(d.x_train),n_output_coeffs+1, figsize=(10, 6))
plt.set_cmap('cubehelix')
plt.tick_params(left = False, bottom=False)
#plt.tight_layout()
fig.subplots_adjust(hspace=0, wspace= 0, bottom = 0.01, left = 0.1, top = 0.7,  right = 0.9)

for idx, gal in enumerate(d.x_train):

    if idx == 0:
        v = axs[idx,0].set_title("Input", fontsize=10)
        v.set_rotation(70)
        for i,f in enumerate(feature_labels):
            print(idx, len(d.x_train), i+1, n_output_coeffs+1)
            vi = axs[idx,i+1].set_title(f, fontsize=10,ha = 'left')
            vi.set_rotation(70)
    axs[idx,0].imshow(gal)
    axs[idx,0].set_yticklabels([])
    axs[idx,0].set_yticks([])
    axs[idx,0].set_xticks([])
    axs[idx,0].set_xticklabels([])
    #h = axs[idx,0].set_ylabel( d.label_list[d.x_train[idx]],fontsize=10,loc = 'top')
    h = axs[idx,0].set_ylabel( d.label_list[d.y_train[idx]],fontsize=10, ha = 'right')
    h.set_rotation(0)    

    for i,f in enumerate(range(n_output_coeffs)):
        axs[idx,i+1].imshow(feature_matrix[idx,f,:,:])
        axs[idx,i+1].set_yticklabels([])
        axs[idx,i+1].set_xticklabels([])
        axs[idx,i+1].set_yticks([])
        axs[idx,i+1].set_xticks([])

plt.show()



