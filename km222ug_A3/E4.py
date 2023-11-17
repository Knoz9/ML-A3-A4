import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = np.genfromtxt("data/bm.csv", delimiter=",")


x, y = data[:, :-1], data[:, -1]

num_samples, train_size = data.shape[0], round(int(0.9 * len(data)))
test_size = num_samples - train_size
indices = np.random.permutation(num_samples)


x_train, yTrain = x[indices[:train_size]], y[indices[:train_size]]
x_test, y_test = x[indices[:test_size]], y[indices[:test_size]]

n_trees = 100
bootstrap_size = 5000
forest = []

for _ in range(n_trees):
    sample_indices = np.random.choice(len(x_train), size=bootstrap_size, replace=True)
    x_samples = x_train[sample_indices].copy()
    y_samples = yTrain[sample_indices].copy()
    tree = DecisionTreeClassifier()
    tree.fit(x_samples, y_samples)
    forest.append(tree)


def prediction(x_test):
    predictions = [tree.predict(x_test) for tree in forest]
    combined_pred = np.round(np.mean(predictions, axis=0))
    return combined_pred

accuracies = [accuracy_score(y_test, tree.predict(x_test)) for tree in forest]
    
y_pred = prediction(x_test)
accuracy = accuracy_score(y_test, y_pred)
generalization_error = 1 - accuracy
average_error = 1 - np.mean(accuracies)

print("***** Performance Summary *****")
print("Generalization Estimate: {:.4f}".format(generalization_error))
print("Average Error of Trees: {:.4f}".format(average_error))


x_min, x_max = np.min(x[:, 0]) - 1, np.max(x[:, 0]) + 1
y_min, y_max = np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

fig, axes = plt.subplots((len(forest) + 9) // 10, 10, figsize=(12, 8), sharex=True, sharey=True)

for i, tree in enumerate(forest):
    row = i // 10
    col = i % 10
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axes[row, col].contourf(xx, yy, Z, cmap=ListedColormap(['blue', 'red']))
    axes[row, col].set_title(f'T = {i+1}')

plt.tight_layout()

plt.figure()
Z = prediction(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=ListedColormap(['blue', 'red']))
plt.title('Ensemble')

plt.show()

