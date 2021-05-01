from scipy import ndimage
import cv2
import numpy as np
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt

def gaussian_blur(size=5, sigma=1.4):

    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def gradient_calculation(img, Dx, Dy):

    Ix = ndimage.filters.convolve(img, Dx)
    Iy = ndimage.filters.convolve(img, Dy)


    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255


    theta = np.arctan2(Iy, Ix)

    return (G, theta)


def thresholding(G, threshold):

    M, N = G.shape
    result = np.zeros((M,N), dtype=np.int32)

    i, j = np.where(G >= threshold)


    result[i, j] = 255

    return result

def general(img, size, sigma, Dx, Dy, threshold):

    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    S_I = convolve(gray, gaussian_blur(size, sigma))

    G, theta = gradient_calculation(S_I, Dx, Dy)

    result = thresholding(G, threshold)

    return result

def visualize(img, format=None, gray=False):

    if img.shape[0] == 3:
        img = img.transpose(1,2,0)

    plt.imshow(img, format)
    plt.show()


name = input("Enter input image name: ")
size = int(input("Enter kernel size: "))
sigma = float(input("Enter sigma value: "))
threshold = int(input("Enter threshold value: "))

Dx = []

Dy = []

for i in range(3):
    col = []
    for j in range(3):
        col.append(float(input("Enter derivative mask dx element at: " + str(i) + "-" + str(j) + ": ")))

    Dx.append(col)

Dx = np.array(Dx)

for i in range(3):
    col = []
    for j in range(3):
        col.append(float(input("Enter derivative mask dy element at: " + str(i) + "-" + str(j) + ": ")))

    Dy.append(col)

Dy = np.array(Dy)

I = cv2.imread(name)

T_I = general(I, size, sigma, Dx, Dy, threshold)

visualize(T_I)

cv2.imwrite("general_output.jpeg", T_I)
