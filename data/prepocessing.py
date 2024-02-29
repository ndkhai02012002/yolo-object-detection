import numpy as np
import cv2
import config

def resize_image(image, size):
    image_resized = cv2.resize(image, size)
    return image_resized

def normalization_image(image):
    return image/255.0

def onehotencoder(label):
    onehot = np.zeros(int(config.C))
    onehot[label] = 1
    return onehot
