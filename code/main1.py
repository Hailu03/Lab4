from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()

# divide dataset into features and labels
X = iris.data
y = iris.target

# create model
model = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)

# predict output
predictions = model.predict(X)

# print accuracy
print("Accuracy is: ", round(accuracy_score(y, predictions)*100,2),"%")