from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Generate the "moons" dataset
X_moons, y_moons = make_moons(n_samples=1000, noise=0.05)

# Scale the features
scaler = StandardScaler()
X_moons = scaler.fit_transform(X_moons)

# Apply k-means
kmeans_moon = KMeans(n_clusters=2, random_state=42)
kmeans_moon.fit(X_moons)

# Apply DBSCAN
dbscan_moon = DBSCAN(eps=0.3, min_samples=5)
dbscan_moon.fit(X_moons)

# Plot k-means results
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=kmeans_moon.labels_, cmap='viridis')
plt.title('K-means Clustering (moon)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot DBSCAN results
plt.subplot(122)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=dbscan_moon.labels_, cmap='viridis')
plt.title('DBSCAN Clustering (moon)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()