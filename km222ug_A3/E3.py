import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = np.genfromtxt("data/mnist_train.csv", delimiter=",", skip_header=1)
y, x = data[:, 0], data[:, 1:]

train_size = 10000
test_size = 1000

x_train, x_val = x[:train_size], x[train_size:train_size+test_size]
y_train, y_val = y[:train_size], y[train_size:train_size+test_size]

param_grid = {"C_values": [1, 5, 10], "gamma_values": [1e-06, 1e-05, 1e-03]}
# I searched for the best parameters of these ones, and found C=5 and gamma = 1e-06.
# I found this out using GridSearch with the paramgrid shown.
# I have removed this code since we now know the best hyperparameters.

kernel_type = "rbf"
gamma_type = "scale"
C_value = 5

svm_classifier = SVC(kernel=kernel_type, gamma=gamma_type, C=C_value)
svm_classifier.fit(x_train, y_train)

predicted_labels = svm_classifier.predict(x_val)
accuracy_percentage = accuracy_score(y_val, predicted_labels) * 100
print("OVO Accuracy: {:.2f}%".format(accuracy_percentage))


def OVA(X, y, C):
    classes = np.unique(y)
    models = {class_label: SVC(kernel=kernel_type, C=C, gamma=gamma_type, probability=True).fit(X, np.where(y == class_label, 1, 0)) for class_label in classes}
    return models


ova_models = OVA(x_train, y_train, C=5)
num_classes = len(ova_models)
num_samples = len(x_val)
probabilities = np.zeros((num_samples, num_classes))

for class_index, class_label in enumerate(ova_models):
    model = ova_models[class_label]
    binary_labels = np.where(y_val == class_label, 1, 0)
    probabilities[:, class_index] = model.predict_proba(x_val)[:, 1]

predicted_labels = np.argmax(probabilities, axis=1)
accuracy = np.mean(predicted_labels == y_val) * 100

print("OVA Accuracy: {:.2f}%".format(accuracy))