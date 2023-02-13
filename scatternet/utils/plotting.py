import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler

def plot_features(x0, x_in, y_in, feature_matrix, indices, feature_labels,label_list):
    n_output_coeffs = feature_matrix.shape[1]
    plt.clf()
    fig, axs = plt.subplots(len(indices),n_output_coeffs+2, figsize=(10, 6))
    plt.tick_params(left = False, bottom=False)
    #plt.tight_layout()
    fig.subplots_adjust(hspace=0, wspace= 0, bottom = 0.01, left = 0.25, top = 0.7,  right = 0.99)

    for idx, gal in enumerate(indices):

        if idx == 0:
            v = axs[idx,0].set_title("Original input image", fontsize=10)
            v1 = axs[idx,1].set_title("Cleaned input image", fontsize=10)
            v.set_rotation(70)
            v1.set_rotation(70)
            for i,f in enumerate(feature_labels):
                vi = axs[idx,i+2].set_title(f, fontsize=10)
                vi.set_rotation(70)
        axs[idx,0].imshow(x0[gal,:,:])
        axs[idx,0].set_yticklabels([])
        axs[idx,0].set_xticklabels([])
        h = axs[idx,0].set_ylabel( label_list[y_in[gal]],fontsize=10,loc = 'top')
        h.set_rotation(-10)

        axs[idx,1].imshow(x_in[gal,:,:])
        axs[idx,1].set_yticklabels([])
        axs[idx,1].set_xticklabels([])
        

        for i,f in enumerate(range(n_output_coeffs)):
            axs[idx,i+2].imshow(feature_matrix[gal,f,:,:])
            axs[idx,i+2].set_yticklabels([])
            axs[idx,i+2].set_xticklabels([])

    plt.show()


def bench_k_means(estimator, data, labels):

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results = [m(labels, estimator.labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator.labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "Homogeneity: {:.3f}; Completeness: {:.3f}; V Measure: {:.3f};\n Adjusted rand: {:.3f}; Adjusted mutual info: {:.3f}; Silhouette: {:.3f}"
    )
    return (formatter_result.format(*results))

def make_pca_plot(feature_matrix, y_test,titlestr, outsr ):
    print(feature_matrix.shape)
    feature_matrix = StandardScaler().fit_transform(feature_matrix.reshape(feature_matrix.shape[0],-1))
    reduced_data = PCA(2).fit_transform(feature_matrix)
    reduced_data = np.array(reduced_data, dtype=np.double)
    kmeans = KMeans(init="k-means++", n_clusters=n_classes, n_init=4)
    estimator = kmeans.fit(reduced_data)
    #model.evaluate(x_test, y_test)

    score_str = bench_k_means(estimator, reduced_data,y_test)
    print(score_str)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )

    import itertools
    marker = itertools.cycle((',', '+', '.', 'o', '*')) 
    for k in keys:
        k_reduced_data = reduced_data[np.where(y_test==k)]
        plt.plot(k_reduced_data[:, 0], k_reduced_data[:, 1],  marker = next(marker), markersize=2, linestyle='')
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )

    title =  titlestr+"\nCentroids are marked with white cross\n"+score_str
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(outsr)
    #plt.show()

def make_svm_plot(X, y, target_names, titlestr, outsr ):
    n_features = X.shape[1]*X.shape[2]*X.shape[3]
    print(n_features)
    feature_matrix = StandardScaler().fit_transform(X.reshape(X.shape[0],-1))
    X_pca = PCA(min(n_features/2., 50)).fit_transform(feature_matrix)

    print(X_pca.shape)

    clf = svm.SVC()
    clf.fit(X_pca, y)

    #t0 = time()
    y_pred = clf.predict(X_pca)
    #print("done in %0.3fs" % (time() - t0))

    print(metrics.classification_report(y, y_pred, target_names=target_names))
    metrics.ConfusionMatrixDisplay.from_estimator(
        clf, X_pca, y, display_labels=target_names, xticks_rotation="vertical"
    )
    plt.tight_layout()
    plt.show()