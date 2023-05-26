import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)

# Evaluate accuracy
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
accuracy = accuracy_score(y, labels)

print("DBSCAN Accuracy:", accuracy)

# Define range of eps and min_samples values
eps_values = np.linspace(0.1, 2.0, num=20)
min_samples_values = range(2, 10)

accuracy_results = np.zeros((len(eps_values), len(min_samples_values)))

# Perform DBSCAN with different eps and min_samples values
for i, eps in enumerate(eps_values):
    for j, min_samples in enumerate(min_samples_values):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X)
        labels = dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        accuracy = accuracy_score(y, labels)
        accuracy_results[i, j] = accuracy

# Visualize the accuracy results
fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.matshow(accuracy_results, cmap='viridis')
fig.colorbar(cax)
ax.set_xticks(np.arange(len(min_samples_values)))
ax.set_yticks(np.arange(len(eps_values)))
ax.set_xticklabels(min_samples_values)
ax.set_yticklabels(["{:.1f}".format(eps) for eps in eps_values])
ax.set_xlabel('min_samples')
ax.set_ylabel('eps')
ax.set_title('Accuracy with Different eps and min_samples')
plt.show()

# Calculate WSS values
wss_results = np.zeros((len(eps_values), len(min_samples_values)))

for i, eps in enumerate(eps_values):
    for j, min_samples in enumerate(min_samples_values):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X)
        labels = dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            wss = 0
            for k in range(n_clusters):
                cluster_points = X[labels == k]
                cluster_center = np.mean(cluster_points, axis=0)
                wss += np.sum((cluster_points - cluster_center) ** 2)
            wss_results[i, j] = wss

# Visualize the WSS results
fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.matshow(wss_results, cmap='viridis')
fig.colorbar(cax)
ax.set_xticks(np.arange(len(min_samples_values)))
ax.set_yticks(np.arange(len(eps_values)))
ax.set_xticklabels(min_samples_values)
ax.set_yticklabels(["{:.1f}".format(eps) for eps in eps_values])
ax.set_xlabel('min_samples')
ax.set_ylabel('eps')
ax.set_title('Within-Cluster-Sum of Squared Errors (WSS) with Different eps and min_samples')
plt.show()