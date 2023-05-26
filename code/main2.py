from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()

# divide dataset into features and labels
X = iris.data
y = iris.target

scores = []
wss = []
ks = range(2, 11)

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=40, n_init="auto").fit(X)

    pred = kmeans.predict(X)
    score = round(accuracy_score(y, pred), 4)
    scores.append(score)
    wss.append(kmeans.inertia_)

plt.plot(ks, scores, color='purple')
plt.xlabel('Number of clusters')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of clusters')
plt.show()

plt.plot(ks, wss,color='red')
plt.xlabel('k')
plt.ylabel('Within-Cluster-Sum of Squared Errors')
plt.title('Within-Cluster-Sum of Squared Errors with different k')
plt.show()
