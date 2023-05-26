import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):
        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        self.centroids = [random.choice(X_train)]
        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
            # Normalize the distances
            dists /= np.sum(dists)
            # Choose remaining points based on their distances
            new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
            self.centroids += [X_train[new_centroid_idx]]
        # This initial method of randomly selecting centroid starts is less effective
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]
        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs
    
    def calculate_wcss(self, X):
        wcss = 0
        for i in range(self.n_clusters):
            cluster_points = X[np.where(np.array(classification) == i)]
            centroid = self.centroids[i]
            cluster_error = np.sum((cluster_points - centroid) ** 2)
            wcss += cluster_error
        return wcss
    
# Load dataset
iris = load_iris()

# divide dataset into features and labels
X = iris.data
y = iris.target

clusters = np.arange(2, 11)

scores = []
wss = []

for center in clusters:
    kmeans = KMeans(n_clusters=center)
    kmeans.fit(X)
    true_labels = y
    class_clusters, classification = kmeans.evaluate(X)
    print(f"Accuracy with clusters = {center} is: ", round(accuracy_score(y, classification)*100,2),"%")

    scores.append(round(accuracy_score(y, classification)*100,2))
    wss.append(kmeans.calculate_wcss(X))

plt.plot(clusters, scores)
plt.xlabel("Number of clusters")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of clusters")
plt.show()

plt.plot(clusters, wss)
plt.xlabel("Number of clusters")
plt.ylabel("Within cluster sum of squares")
plt.title("Within cluster sum of squares vs Number of clusters")
plt.show()


