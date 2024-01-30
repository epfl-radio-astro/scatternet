from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def check_classifier(clf, X, y, label_list,title = ''):

    #t0 = time()
    y_pred = clf.predict(X)
    #print("done in %0.3fs" % (time() - t0))

    print(metrics.classification_report(y, y_pred, target_names= [str(l) for l in label_list]))
    metrics.ConfusionMatrixDisplay.from_estimator(
        clf, X, y, display_labels=label_list, xticks_rotation="vertical", normalize = 'true'
    )
    acc = metrics.balanced_accuracy_score(y, y_pred)
    plt.title("{0}\nBalanced accuracy = {1:0.2f}".format(title,acc))
    plt.tight_layout()
    plt.show()

class GenericClassifier():
    '''
    Abstract class that defines the functionality of Classifiers in this library
    '''
    _estimator_type = "classifier"
    def __init__(self, clf, doPCA=False, n_components_pca = None): 
        self.clf = clf
        self.pca = None
        if doPCA:
            if n_components_pca == None:
                raise RuntimeError("PCA enabled but # of components not set.")
            self.pca = PCA(n_components_pca)


    def fit(self,x,y, x_val=None, y_val=None, verbose = False):
        x = x.reshape(x.shape[0],-1)
        self.n_features = x.shape[-1]
        self.scaler = StandardScaler()
        self.x = self.scaler.fit_transform(x)
        if self.pca:
            self.x = self.pca.fit_transform(self.x)
        #self._fit_clf(self.x,y)
        self.clf.fit(self.x, y)

    def predict(self,x):
        x = x.reshape(x.shape[0],-1)
        x = self.scaler.transform(x)
        if self.pca:
            x = self.pca.transform(x)
        return self.clf.predict(x)

    def from_predictions(self,**kwargs):
        return self.cls.from_predictions(**kwargs)

class ClassifierSVC(GenericClassifier):
    _estimator_type = "classifier"
    def __init__(self, doPCA=False, n_components_pca = None): 
        clf = svm.SVC(kernel="linear", class_weight="balanced")
        super().__init__(clf, doPCA, n_components_pca)

class ClassifierRandomForest(GenericClassifier):
    _estimator_type = "classifier"
    def __init__(self, doPCA=False, n_components_pca = None): 
        clf = RandomForestClassifier(max_depth=3, random_state=0)
        super().__init__(clf, doPCA, n_components_pca)

class ClassifierNN():
    _estimator_type = "classifier"
    def __init__(self, model, dataset, outdir = "./", outname = "model_checkpoint"): 
        self.model = model
        self.dataset = dataset
        self.outdir = outdir
        self.outname = outname
        model.compile(optimizer='adam',
              loss='binary_crossentropy' if len(dataset.label_list) ==2 else 'categorical_crossentropy',
              metrics=['accuracy'], weighted_metrics = ['accuracy'])

    def predict(self,x):
        return self.dataset.decode(self.model.predict(x))

    def fit(self,x,y, x_val, y_val, verbose = 0):
        checkpoint_name = self.outdir + self.outname
        mcp_save = tf.keras.callbacks.ModelCheckpoint(checkpoint_name,
                                                      save_weights_only=True,
                                                      save_best_only=True,
                                                      monitor='val_accuracy', mode='max', verbose = verbose)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            verbose=verbose,
            patience=10,
            mode='max',
            restore_best_weights=True)

        self.model.fit(x,
                  self.dataset.encode(y),
                  epochs=50, batch_size=32,
                  validation_data = (x_val, self.dataset.encode(y_val)),
                  sample_weight=self.dataset.sample_weights,
                  callbacks = [early_stopping, mcp_save], verbose = verbose)
        self.model.load_weights(checkpoint_name)
