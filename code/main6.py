from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Generate circles dataset
X_circles, y_circles = make_circles(n_samples=1000, factor=0.5, noise=0.05)

# Scale the features
scaler = StandardScaler()
X_circles = scaler.fit_transform(X_circles)

# Apply k-means
kmeans_circles = KMeans(n_clusters=2, random_state=42)
kmeans_circles.fit(X_circles)

# Apply DBSCAN
dbscan_circles = DBSCAN(eps=0.3, min_samples=5)
dbscan_circles.fit(X_circles)

# Plot k-means results
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=kmeans_circles.labels_, cmap='viridis')
plt.title('K-means Clustering (circles)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot DBSCAN results
plt.subplot(122)
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=dbscan_circles.labels_, cmap='viridis')
plt.title('DBSCAN Clustering (circles)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
