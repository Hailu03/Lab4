from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Generate blobs dataset
X_blobs, y_blobs = make_blobs(n_samples=1000, centers=3, cluster_std=1.5)

# Scale the features
scaler = StandardScaler()
X_blobs = scaler.fit_transform(X_blobs)

# Apply k-means
kmeans_blobs = KMeans(n_clusters=3, random_state=42)
kmeans_blobs.fit(X_blobs)

# Apply DBSCAN
dbscan_blobs = DBSCAN(eps=0.3, min_samples=5)
dbscan_blobs.fit(X_blobs)

# Plot k-means results
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=kmeans_blobs.labels_, cmap='viridis')
plt.title('K-means Clustering (blobs)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot DBSCAN results
plt.subplot(122)
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=dbscan_blobs.labels_, cmap='viridis')
plt.title('DBSCAN Clustering (blobs)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()