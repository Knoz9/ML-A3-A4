import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA


def sammons_stress(iX, oX):
    in_upper_triangle = np.triu(iX)
    out_upper_triangle = np.triu(oX)
    
    division_result = np.divide(out_upper_triangle - in_upper_triangle, in_upper_triangle, where=in_upper_triangle!=0)
    squared_result = np.square(division_result)
    sum_squared_result = np.sum(squared_result)
    sum_in = 1 / np.sum(in_upper_triangle)
    
    stress = sum_in * sum_squared_result
    return stress


def sammon(X, max_iter, epsilon, alpha):
    distance_matrix = pairwise_distances(X)
    distance_matrix[distance_matrix == 0] = 1e-30
    total_distance = np.sum(np.triu(distance_matrix))
    y_indices = range(X.shape[0])

    Y = make_blobs(n_samples=X.shape[0], n_features=2, centers=1, random_state=1)[0]

    for i in range(max_iter):
        projected_dist = pairwise_distances(Y)
        projected_dist[projected_dist == 0] = 1e-30
        E = sammons_stress(distance_matrix, projected_dist)
        print(f'[Sammons Calc]: I = {i}/{max_iter}')
        if E < epsilon:
            break

        for j in y_indices:
            first, second = np.array([0, 0], dtype=np.float64)
            for k in y_indices:
                if k == j:
                    pass
                first += (
                    (distance_matrix[j, k] - projected_dist[j, k])
                    / (projected_dist[j, k] * distance_matrix[j, k])
                ) * (Y[j] - Y[k])

                second += (
                    1 / (distance_matrix[j, k] * projected_dist[j, k])
                ) * (
                    (distance_matrix[j, k] - projected_dist[j, k])
                    - ((np.square(Y[j] - Y[k]) / projected_dist[j, k]) * (1 + ((distance_matrix[j, k] - projected_dist[j, k]) / projected_dist[j, k])))
                )

            Y[j] = Y[j] - alpha * (
                (-2 / total_distance) * first
            ) / np.abs((-2 / total_distance) * second)
    return Y


def main():
    X, y = make_blobs(500, n_features=5, centers=1, random_state=1)
    Y = sammon(X, max_iter=100, epsilon=0.001, alpha=0.1)
    plt.scatter(Y[:,0], Y[:,1])
    plt.show()

# main()