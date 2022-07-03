import pandas as pd
from sklearn.decomposition import PCA
import cv2

def vectorPCA(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    blue,green,red = cv2.split(img)

    df_blue = blue/255
    df_green = green/255
    df_red = red/255
    
    pcab = PCA(n_components = 50)
    blue_transformed = pcab.fit_transform(df_blue)
    pcag = PCA(n_components = 50)
    green_transformed = pcag.fit_transform(df_green)
    pcar = PCA(n_components = 50)
    red_transformed = pcar.fit_transform(df_red)

    return [blue_transformed,green_transformed,red_transformed,pcab,pcag,pcar]

def reversePCA(rgb):
    
    blue = rgb[3].inverse_transform(rgb[0])
    green = rgb[4].inverse_transform(rgb[1])
    red = rgb[5].inverse_transform(rgb[2])

    img = cv2.merge((blue,green,red))

    return img
