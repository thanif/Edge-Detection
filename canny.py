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


def gradient_calculation(img):


    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)


    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255


    theta = np.arctan2(Iy, Ix)

    return (G, theta)

def non_maximum_suppression(G, theta):

    M, N = G.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180


    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255

               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = G[i, j+1]
                    r = G[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = G[i+1, j-1]
                    r = G[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = G[i+1, j]
                    r = G[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = G[i-1, j-1]
                    r = G[i+1, j+1]

                if (G[i,j] >= q) and (G[i,j] >= r):
                    Z[i,j] = G[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z

def double_thresholding(Z, lowThresholdRatio=0.09, highThresholdRatio=0.17, w_p_v=75):

    highThreshold = Z.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = Z.shape
    result = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(w_p_v)
    strong = np.int32(255)

    strong_i, strong_j = np.where(Z >= highThreshold)
    zeros_i, zeros_j = np.where(Z < lowThreshold)

    weak_i, weak_j = np.where((Z <= highThreshold) & (Z >= lowThreshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return (result, weak, strong)

def edge_tracking(img, weak, strong=255):

    M, N = img.shape

    for i in range(1, M-1):
        for j in range(1, N-1):

            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def canny(img, size, sigma, l_t_r, h_t_r, w_p_v):

    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    S_I = convolve(gray, gaussian_blur(size, sigma))

    G, theta = gradient_calculation(S_I)

    Z = non_maximum_suppression(G, theta)

    result, weak, strong = double_thresholding(Z, l_t_r, h_t_r, w_p_v)

    img = edge_tracking(result, weak, strong)

    return img

def visualize(img, format=None, gray=False):

    if img.shape[0] == 3:
        img = img.transpose(1,2,0)

    plt.imshow(img, format)
    plt.show()


name = input("Enter input image name: ")
size = int(input("Enter kernel size: "))
sigma = float(input("Enter sigma value: "))
l_t_r = float(input("Enter low threshold ratio: "))
h_t_r = float(input("Enter high threshold ratio: "))
w_p_v = int(input("Enter weak pixel intensity value: "))

I = cv2.imread(name)

T_I = canny(I, size, sigma, l_t_r, h_t_r, w_p_v)

visualize(T_I)

cv2.imwrite("canny_output.jpeg", T_I)
