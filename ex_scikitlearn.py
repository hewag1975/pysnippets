from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3],  # 2 samples, 3 features
     [11, 12, 13]]
y = [0, 1]  # classes of each sample
clf.fit(X, y)
clf.predict(X)  # predict classes of the training data

clf.predict([[4, 5, 6], [14, 15, 16]])  # predict classes of new data

from sklearn.preprocessing import StandardScaler
X = [[0, 15],
     [1, -10]]


# scale data according to computed scaling values
StandardScaler().fit(X).transform(X)




from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Create an instance of StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data
scaler.fit(X)

# Transform the data
X_scaled = scaler.transform(X)

# Print the scaled data
print(X_scaled)

