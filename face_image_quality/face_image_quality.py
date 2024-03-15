import cv2

import numpy as np
import os
from numpy.linalg import norm





image_path = cv2.imread(r'G:\datasets\new_dat\1\006439.jpg')
#gray = cv2.imread(r'/home/khlifi/Documents/test_bgg/n000017/53.jpg', 0)


def brightness(img):
    # A good value for brightness would be 80
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)
#print(f"brightness is: {brightness(img)}")
def blurr_detection(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian
def resolution_measurement(img):
    wid = img.shape[1]
    hgt = img.shape[0]
    print(str(wid) + "x" + str(hgt))
    return(wid, hgt)
#resolution_measurement(img)

def check_image_quality(image_path):
    img = cv2.imread(image_path)
    if resolution_measurement(img) == (383,524) and blurr_detection(img)>100 and brightness(img)>90:
        print("image quality is acceptable")
        return 1
    else:
        print('image was not accepted')
        return 0
check_image_quality()

