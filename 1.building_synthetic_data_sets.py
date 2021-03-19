###1###

from sklearn import datasets

# Generating static problem of qualification
X, y = datasets.make_classification(
    n_features=2,           #two attributes
    n_informative=2,        #both informative
    n_repeated=0,           #none repeated and redundant
    n_redundant=0,
    flip_y=.1,              #label noise 10%
    n_samples=300,
    n_clusters_per_class=1, #equally distributed

    random_state=1410,      #seed

)

# Putting labels in the last row
import numpy as np
dataset = np.concatenate((X, y[:, np.newaxis]), axis=1)

#Saving to .csv file
np.savetxt(
    "building_synthetic_data_sets.csv",
    dataset,
    delimiter=",\t",
    fmt=["%.5f" for i in range(X.shape[1])] + ["%i"],
)

# Drawing the plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.xlabel("$x^1$")
plt.ylabel("$x^2$")
plt.tight_layout()
plt.savefig('building_synthetic_data_sets.png')

print("FINISHED")
