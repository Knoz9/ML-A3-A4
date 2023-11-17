import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import seaborn as sns

data = np.genfromtxt("data/fashion-mnist_train.csv", delimiter=',', skip_header=1)
dataTest = np.genfromtxt("data/fashion-mnist_test.csv", delimiter=',', skip_header=1)

labels = data[:, 0]
labels_test = dataTest[:,0]
x = data[:, 1:]
x_test = dataTest[:, 1:]

rand_indices = np.random.choice(len(labels), size=16, replace=False) 

def MLP(hiddenLayers, units, regularization, learningRate):
    model = Sequential()
    model.add(Dense(units=units, activation='relu', input_shape=(784,)))

    for _ in range(hiddenLayers):
        model.add(Dense(units=units, activation='relu'))

    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=MLP)

#Here are the hyperparameters i found using gridsearch.

hidden_layers = [3]
units = [256]
regularization = [0.01]
learning_rate = [0.001]

params = {
    'hiddenLayers': hidden_layers,
    'units': units,
    'regularization': regularization,
    'learningRate': learning_rate
}

fig, axes = plt.subplots(4, 4)
fig.suptitle("Randomly Selected Images")

for i, ax in enumerate(axes.flatten()):
    index = rand_indices[i]
    image = np.reshape(x[index], (28, 28))
    label = labels[index]
    ax.imshow(image, cmap='gray')
    ax.set_title(f"label: {label}")


plt.tight_layout()

grid_search = GridSearchCV(model, params, cv=3, scoring='accuracy')
grid_search.fit(x, labels)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(x_test, labels_test)
print(f"Best Hyperparameters: {best_params}")
print(f"Test Accuracy:  {test_accuracy*100:.2f}%")

fig, ax = plt.subplots()
y_pred = best_model.predict(x_test)
confusion_matrix = confusion_matrix(labels_test, y_pred)
sns.heatmap(confusion_matrix, cmap="Reds", cbar=False,  annot=True, fmt="d", linewidths=0.5, linecolor='black', ax=ax)
ax.set_xlabel("Prediction Labels")
ax.set_ylabel("Real Labels")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()


