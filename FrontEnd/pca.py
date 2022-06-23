from sklearn.decomposition import PCA
import cv2

def vectorPCA(path):
    img = cv2.cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    blue,green,red = cv2.split(img)

    pca = PCA(50)

    blue_transformed = pca.fit_transform(blue)
    green_transformed = pca.fit_transform(green)
    red_transformed = pca.fit_transform(red)

    return [blue_transformed,green_transformed,red_transformed]

def reversePCA(rgb):
    pca = PCA(50)
    blue = pca.inverse_transform(rgb[0])
    green = pca.inverse_transform(rgb[1])
    red = pca.inverse_transform(rgb[2])

    img = cv2.merge(blue,green,red)

    return img
