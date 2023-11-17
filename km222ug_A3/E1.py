import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def grid_search(kernels, X_train, y_train, X_val, y_val):
    best_models = {}
    for kernel, params in kernels.items():
        best_score = 0
        best_model = None
        for param_values in params:
            clf = SVC(kernel=kernel, **param_values)
            clf.fit(X_train, y_train)
            score = clf.score(X_val, y_val)
            if score > best_score:
                best_score = score
                best_model = clf
        best_models[kernel] = best_model
    return best_models

def main():
    with open("data/dist.csv", "r", encoding="utf-8-sig") as file:
        next(file)
        data = np.loadtxt(file, delimiter=";")
    train_stop = round(data.shape[0] * 0.8)
    X, y = data[:, :-1], data[:, -1]
    X_train, y_train = X[:train_stop, :], y[:train_stop]

    with open("data/dist_val.csv", "r", encoding="utf-8-sig") as file:
        next(file)
        val_data = np.loadtxt(file, delimiter=";")
    X_val, y_val = val_data[:, :-1], val_data[:, -1]

    kernels = {
        "linear": [{"C": 0.001}, {"C": 0.01}, {"C": 0.1}, {"C": 1}, {"C": 10}],
        "rbf": [{"C": 0.001, "gamma": 0.001}, {"C": 0.01, "gamma": 0.01}, {"C": 0.1, "gamma": 0.1}],
        "poly": [{"C": 0.01, "degree": 2, "gamma": 0.01}, {"C": 0.1, "degree": 3, "gamma": 0.05}, {"C": 1, "degree": 4, "gamma": 0.1}]
    }

    best_models = grid_search(kernels, X_train, y_train, X_val, y_val)

    margin = 0.5
    grid_size = 500
    x_min, x_max = min(X[:, 0]) - margin, max(X[:, 0]) + margin
    y_min, y_max = min(X[:, 1]) - margin, max(X[:, 1]) + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    grid = np.c_[xx.ravel(), yy.ravel()]

    cmap = "prism"

    plt.figure("Best models", figsize=(16, 6))
    plt.subplot(1, 3, 1)
    plt.gca().set_title(f"Linear\nC={best_models['linear'].C}\nScore: {round(best_models['linear'].score(X_val, y_val), 3)}")
    plt.imshow(best_models['linear'].predict(grid).reshape(xx.shape), origin="lower", extent=(x_min, x_max, y_min, y_max), cmap=cmap)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker=".", cmap=cmap)

    plt.subplot(1, 3, 2)
    plt.gca().set_title(f"Rbf\nC={best_models['rbf'].C}, gamma={best_models['rbf'].gamma}\nScore: {round(best_models['rbf'].score(X_val, y_val), 3)}")
    plt.imshow(best_models['rbf'].predict(grid).reshape(xx.shape), origin="lower", extent=(x_min, x_max, y_min, y_max), cmap=cmap)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker=".", cmap=cmap)

    plt.subplot(1, 3, 3)
    plt.gca().set_title(f"Poly\nC={best_models['poly'].C}, gamma={best_models['poly'].gamma}, d={best_models['poly'].degree}\nScore: {round(best_models['poly'].score(X_val, y_val), 3)}")
    plt.imshow(best_models['poly'].predict(grid).reshape(xx.shape), origin="lower", extent=(x_min, x_max, y_min, y_max), cmap=cmap)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker=".", cmap=cmap)

    plt.tight_layout()
    plt.show()



main()