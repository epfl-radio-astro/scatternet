from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm, metrics
import matplotlib.pyplot as plt

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

class ClassifierSVC():
    _estimator_type = "classifier"
    def __init__(self, doPCA=False, n_components_pca = None): 
        self.pca = None
        if doPCA:
            if n_components_pca == None:
                raise RuntimeError("PCA enabled but # of components not set.")
            self.pca = PCA(n_components)

    def fit(self,x,y):
        print("Fitting classifier with input shape", x.shape)
        x = x.reshape(x.shape[0],-1)
        self.n_features = x.shape[-1]
        self.scaler = StandardScaler()
        self.x = self.scaler.fit_transform(x)
        if self.pca:
            self.x = self.pca.fit_transform(self.x)
        self.clf = svm.SVC(kernel="linear", class_weight="balanced")
        self.clf.fit(self.x, y)

    def predict(self,x):
        print("Predicting classifier with input shape", x.shape)
        x = x.reshape(x.shape[0],-1)
        x = self.scaler.transform(x)
        if self.pca:
            x = self.pca.transform(x)
        return self.clf.predict(x)

    def from_predictions(self,**kwargs):
        return self.cls.from_predictions(**kwargs)

