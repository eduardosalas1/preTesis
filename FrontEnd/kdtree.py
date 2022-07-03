import numpy as np
import pickle
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from pca import vectorPCA, reversePCA

rng = np.random.RandomState(0)
X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
tree = KDTree(X, leaf_size=2)        
s = pickle.dumps(tree)                     
tree_copy = pickle.loads(s)                
dist, ind = tree_copy.query(X[:1], k=6)     
print(ind)  # indices of 3 closest neighbors
print(dist)  # distances to 3 closest neighbors

aux1 = vectorPCA("original.jpeg")
aux2 = reversePCA(aux1)
image_array = np.array(aux2)
print(image_array.ndim) #3 DIMENSION
aux_1 = vectorPCA("taza.jpg")
aux_2 = reversePCA(aux1)
image_array1 = np.array(aux_2)
print(image_array1.ndim) #3 DIMENSION

#print(image_array)
#print(aux2.shape)
arr = np.array([image_array,image_array1])
tree = KDTree(arr, leaf_size=4)
s = pickle.dumps(tree)                     
tree_copy = pickle.loads(s)                
dist, ind = tree_copy.query(X[:1], k=2)     
print(ind)  # indices of 3 closest neighbors
print(dist)  # distances to 3 closest neighbors




