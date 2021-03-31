import numpy as np
import cv2
import matplotlib.pyplot as plt


class Harris_corner_detector(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def detect_harris_corners(self, img):
        ### TODO ####
        # Step 1: Smooth the image by Gaussian kernel
        # - Function: cv2.GaussianBlur (kernel = 3, sigma = 1.5)
        img = cv2.GaussianBlur(img, (3, 3), 1.5)

        # Step 2: Calculate Ix, Iy (1st derivative of image along x and y axis)
        # - Function: cv2.filter2D (kernel = [[1.,0.,-1.]] for Ix or [[1.],[0.],[-1.]] for Iy)
        Ix = cv2.filter2D(img, -1, kernel=np.array([[1.,0.,-1.]]))
        Iy = cv2.filter2D(img, -1, kernel=np.array([[1.],[0.],[-1.]]))

        # Step 3: Compute Ixx, Ixy, Iyy (Ixx = Ix*Ix, ...)
        Ixx = Ix ** 2
        Ixy = Ix * Iy
        Iyy = Iy ** 2

        # Step 4: Compute Sxx, Sxy, Syy (weighted summation of Ixx, Ixy, Iyy in neighbor pixels)
        # - Function: cv2.GaussianBlur (kernel = 3, sigma = 1.)
        Sxx = cv2.GaussianBlur(Ixx, (3, 3), 1.)
        Sxy = cv2.GaussianBlur(Ixy, (3, 3), 1.)
        Syy = cv2.GaussianBlur(Iyy, (3, 3), 1.)

        # Step 5: Compute the det and trace of matrix M (M = [[Sxx, Sxy], [Sxy, Syy]])
        det = Sxx * Syy - Sxy ** 2
        trace = Sxx + Syy

        # Step 6: Compute the response of the detector by det/(trace+1e-12)
        response = det / (trace + 1e-12)

        return response
    
    def post_processing(self, response):
        ### TODO ###
        # Step 1: Thresholding
        response = (response > self.threshold).astype(int) * response

        # Step 2: Find local maximum
        local_max = []
        for i in range(response.shape[0]):
            for j in range(response.shape[1]):
                if response[i, j] == 0:
                    continue
                is_local_max = True
                for x in range(i-2 if i-2>=0 else 0, i+3 if i+2<response.shape[0] else response.shape[0]):
                    for y in range(j-2 if j-2>=0 else 0, j+3 if j+2<response.shape[1] else response.shape[1]):
                        if response[i, j] < response[x, y]:
                            is_local_max = False
                            break
                    if is_local_max == False:
                        break
                if is_local_max:
                    local_max.append([i, j])
        
        return local_max