from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from E2 import sammon
from E1 import bkmeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

sammons_maxiter = 100
e = 0.001
A = 0.3

# Figure 1
fig1, axs1 = plt.subplots(3, 3, figsize=(10, 10))
print(f"Max iter set to {sammons_maxiter}. It might take a total of {sammons_maxiter*3} to run the code. Each iteration takes approx 0.5-2 seconds.")
for i in range(3):
    if i == 0:
        datasetX, datasetLabels = load_breast_cancer().data, load_breast_cancer().target
        dataset_name = 'Breast Cancer Dataset'
    elif i == 1:
        datasetX, datasetLabels = load_wine().data, load_wine().target
        dataset_name = 'Wine Dataset'
    elif i == 2:
        diabetesData = np.genfromtxt('csv_result-diabetes.csv', delimiter=',', skip_header=1)
        datasetX, datasetLabels = diabetesData[:, :-1], diabetesData[:, -1]
        dataset_name = 'Diabetes Dataset'
    
    for j in range(3):
        if j == 0:
            print(f"Computing sammon mapping for {dataset_name} ({i+1}/3)")
            result = sammon(datasetX, sammons_maxiter, e, A)
            plot_title = 'Sammon Mapping'
        elif j == 1:
            pca = PCA(n_components=2)
            result = pca.fit_transform(datasetX)
            plot_title = 'PCA'
        elif j == 2:
            tsne = TSNE(n_components=2)
            result = tsne.fit_transform(datasetX)
            plot_title = 't-SNE'
        
        axs1[i, j].scatter(result[:, 0], result[:, 1], c=datasetLabels, s=5)
        axs1[i, j].set_title(f'{dataset_name} - {plot_title}')

plt.tight_layout()

fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i in range(3):
    if i == 0:
        datasetX, datasetLabels = load_breast_cancer().data, load_breast_cancer().target
        dataset_name = 'Breast Cancer Dataset'
    elif i == 1:
        datasetX, datasetLabels = load_wine().data, load_wine().target
        dataset_name = 'Wine Dataset'
    elif i == 2:
        diabetesData = np.genfromtxt('csv_result-diabetes.csv', delimiter=',', skip_header=1)
        datasetX, datasetLabels = diabetesData[:, :-1], diabetesData[:, -1]
        dataset_name = 'Diabetes Dataset'

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(datasetX)

    for j in range(3):
        ax = axes[i, j]
        if j == 0:
            bkmeans_result = bkmeans(X_pca, 2, 30)
            c = bkmeans_result
            algorithm_name =  'bk means'
        elif j == 1:
            kmeans_result = KMeans(n_init = 10, n_clusters=2, random_state=0).fit_predict(X_pca)
            c = kmeans_result
            algorithm_name =  'classic k means'
        elif j == 2:
            hierarchical_result = linkage(X_pca, method='ward')
            dendro = dendrogram(hierarchical_result, ax=ax)
            c = datasetLabels
            algorithm_name =  'hierarchical'

        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=c)
        ax.set_title(f'{dataset_name} - {algorithm_name}')

plt.tight_layout()
plt.show()
