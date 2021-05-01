from scipy.ndimage.filters import convolve
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def calculate_log(x, y, sigma):

	nom = ( (y**2)+(x**2)-2*(sigma**2) )
	denom = ( (2*math.pi*(sigma**6) ))
	expo = math.exp( -((x**2)+(y**2))/(2*(sigma**2)) )
	return nom*expo/denom


def create_log_mask(sigma, size = 7):

	w = math.ceil(float(size)*float(sigma))

	if(w%2 == 0):

		w = w + 1

	log_mask = []

	w_range = int(math.floor(w/2))

	for i in range(-w_range, w_range+1):
		for j in range(-w_range, w_range+1):
			log_mask.append(calculate_log(i,j,sigma))
	log_mask = np.array(log_mask)
	log_mask = log_mask.reshape(w,w)
	return log_mask

def detect_zero_crossing(L_I):

    threshold = np.absolute(L_I).mean() * 0.75

    output = np.zeros(L_I.shape)

    w = output.shape[1]
    h = output.shape[0]

    for y in range(1, h - 1):
        for x in range(1, w - 1):

            patch = L_I[y-1:y+2, x-1:x+2]
            p = L_I[y, x]
            maxP = patch.max()
            minP = patch.min()
            if (p > 0):
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if maxP > 0 else False
            if ((maxP - minP) > threshold) and zeroCross:
                output[y, x] = 255

    return output

def visualize(img, format=None, gray=False):

    if img.shape[0] == 3:
        img = img.transpose(1,2,0)

    plt.imshow(img, format)
    plt.show()

name = input("Enter input image name: ")
size = int(input("Enter kernel size: "))
sigma = float(input("Enter sigma value: "))

I = cv2.imread(name)

r, g, b = I[:,:,0], I[:,:,1], I[:,:,2]
gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

log_mask = create_log_mask(sigma, size)
L_I = convolve(gray, log_mask)

output = detect_zero_crossing(L_I)

visualize(output)

cv2.imwrite("log_output.jpeg", output)
