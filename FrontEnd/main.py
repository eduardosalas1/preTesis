from pca import vectorPCA, reversePCA
from PIL import Image
import numpy as np
import cv2

def main():
    aux1 = vectorPCA("original.jpeg")
    aux2 = reversePCA(aux1)
    image_array = np.array(aux2)
    print(image_array)
    print(aux2.shape)


    print("-------------------------------------------------")

    img = cv2.imread("original.jpeg")
    image_array = np.array(img)
    print(image_array)
    print(img.shape)

    return

if __name__ == "__main__":
    main()