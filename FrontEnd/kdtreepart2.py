from sklearn import datasets# Import the built-in dataset module
from sklearn.neighbors import KNeighborsClassifier# Import the KNN class in the sklearn.neighbors module
import numpy as np
from sklearn.neighbors import KDTree#import KD tree class

np.random.seed(0)# Set the random seed. If it is not set, the system time will be used as the parameter by default. Therefore, the random number generated every time the random module is called is different.
iris = datasets.load_iris()# Import the iris data set, iris is something similar to a structure, with sample data inside, and label data for supervised learning
iris_x = iris.data# Sample data 150*4 two-dimensional data, representing 150 samples, 4 attributes of each sample are the length and width of petals and calyx respectively
iris_y = iris.target# The length 150 is an array, the label of the sample data
indices = np.random.permutation(len(iris_x)) # permutation receives a number as a parameter (150) and generates a one-dimensional array of 0-149, which is just randomly scrambled. Of course, she can also receive a one-dimensional array As a parameter, the result is to directly scramble this array
iris_x_train = iris_x[indices[:-10]]# randomly select 140 samples as the training data set
iris_y_train = iris_y[indices[:-10]]# and select the labels of these 140 samples as the labels of the training data set
iris_x_test = iris_x[indices[-10:]]# The remaining 10 samples are used as test data sets
iris_y_test = iris_y[indices[-10:]]# and use the labels corresponding to the remaining 10 samples as test data and labels

knn = KNeighborsClassifier()# Define a knn classifier object
knn.fit(iris_x_train, iris_y_train)# Call the training method of the object, which mainly receives two parameters: the training data set and its sample label
iris_y_predict = knn.predict(iris_x_test)# Call the test method of the object, which mainly receives one parameter: test data set
probility = knn.predict_proba(iris_x_test)# Calculate the probability-based prediction of each test sample
neighborpoint = knn.kneighbors(iris_x_test[0].reshape(1,-1),n_neighbors=1,return_distance=True)# Calculate the distance from the last test sample at the nearest 5 points, and return the sequence number composition of these samples Array of
score = knn.score(iris_x_test, iris_y_test, sample_weight=None)# Call the scoring method of the object to calculate the accuracy rate

tree = KDTree(iris_x_train)
# ind: Index of the nearest 3 neighbors
# dist: 3 nearest neighbors
X = iris_x_train[0].reshape(1,-1)
dist, ind = tree.query(X, k=3)

print ('ind:',ind)
print ('dist:',dist)



# np.random.seed(0)
# X = np.array([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)])
# tree = KDTree(X, leaf_size=2)