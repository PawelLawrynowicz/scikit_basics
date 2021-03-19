###2###

import numpy as np

# Reading data
dataset = np.genfromtxt("building_synthetic_data_sets", delimiter=",\t")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

# Initializing gauss naive Bayes classificator
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.25,
    random_state=1410,
)

# Training Bayes classificator
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)

# Support matrix
class_probabilities = clf.predict_proba(X_test)
# Calculating predictions
predict = np.argmax(class_probabilities, axis=1)
#print("True labels:     ", y_test)
#print("Predicted labels:", predict)

# Accuracy metric
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, predict)
print("Wartość metryki accuracy:\t %.2f" % score)

# Drawing the plot
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("Etykiety rzeczywiste")
ax2.set_title("Etykiety predykcji")
ax1.set(xlabel='$x^1$', ylabel='$x^2$')
ax2.set(xlabel='$x^1$')
ax1.scatter(class_probabilities[:, 0], class_probabilities[:, 1], c=y_test, cmap='bwr', s=10)
ax2.scatter(class_probabilities[:, 0], class_probabilities[:, 1], c=predict, cmap='bwr', s=10)
plt.savefig('zad1-2.png')