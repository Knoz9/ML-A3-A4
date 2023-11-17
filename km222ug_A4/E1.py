import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import k_means

def bkmeans(X, k, i):
    num_samples = X.shape[0]
    cluster_assignments = [0] * num_samples
    kk = 1
    while kk < k:
        unique_values, value_counts = np.unique(cluster_assignments, return_counts=True)
        common_k = unique_values[value_counts.argmax()]
        common_indices = np.where(np.array(cluster_assignments) == common_k)[0]
        sub_samples = X[common_indices]
        new_clusters = k_means(sub_samples, 2, n_init=i)[1]

        for j, idx in enumerate(common_indices):
            cluster_assignments[idx] = common_k if new_clusters[j] == 0 else kk

        kk += 1

    return np.array(cluster_assignments)

def main():
    X, y = make_blobs(n_samples=1000, centers=5, n_features=5) # Synthetic dataset
    c = bkmeans(X, 5, 100)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=c, cmap='Spectral')
    plt.show()
    
# main()