from time import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering


def sort_labels(labels, descending=True):
    labels = labels.copy()
    isin = labels >= 0
    labels_uniq, labels[isin], counts = np.unique(
            labels[isin], return_inverse=True, return_counts=True)
    c = counts
    if descending:
        c = c * (-1)
    order = c.argsort()
    rank = order.argsort()
    labels[isin] = rank[labels[isin]]
    return labels, labels_uniq[order]


def cluster(x, n_clusters, method='mbkm', sort_by_frequency=True):

    mask = np.isfinite(x).all(-1)
    x = x[mask]
    print(f'Clustering pixels using {method}...')
    t0 = time()
    if method == 'mbkm':
        model = MiniBatchKMeans(
                n_clusters=n_clusters,
                # batch_size=x.shape[0]//10, max_iter=1000,
                # max_no_improvement=100, n_init=10,
                random_state=0, verbose=0)
    elif method == 'km':
        model = KMeans(
                n_clusters=n_clusters,
                random_state=0, verbose=0)
    # elif method == 'hdbscan':
    #     min_cluster_size = min(1000, x.shape[0] // 400 + 1)
    #     min_samples = min_cluster_size // 10 + 1
    #     model = HDBSCAN(
    #             min_cluster_size=min_cluster_size,
    #             min_samples=min_samples,
    #             core_dist_n_jobs=64)
    elif method == 'agglomerative':
        # knn_graph = kneighbors_graph(x, n_neighbors=10, include_self=False)
        model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward', compute_distances=True)
    else:
        raise ValueError(f'Method `{method}` not recognized')
    print(x.shape)
    labels = model.fit_predict(x)
    print(int(time() - t0), 'sec')
    print('n_clusters:', np.unique(labels).size)

    if sort_by_frequency:
        labels = sort_labels(labels)[0]

    labels_arr = np.full(mask.shape, labels.min()-1, dtype=int)
    labels_arr[mask] = labels

    return labels_arr, model
