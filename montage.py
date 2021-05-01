import cv2
from imutils import build_montages

i1 = cv2.imread("lena.jpeg")
i2 = cv2.imread("general_output.jpeg")

images = []

images.append(i1)
images.append(i2)

montages = build_montages(images, (128, 196), (2, 1))

cv2.imwrite("general_montage.png", montages[0])
