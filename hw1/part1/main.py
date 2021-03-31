import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt
from HCD import Harris_corner_detector


def main():
    parser = argparse.ArgumentParser(description='main function of Harris corner detector')
    parser.add_argument('--threshold', default=100., type=float, help='threshold value to determine corner')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    ### TODO ###
    HCD = Harris_corner_detector(args.threshold)

    response = HCD.detect_harris_corners(img_gray)
    result = HCD.post_processing(response)

    for x, y in result:
        for i in range(x-1 if x-1>=0 else x, x+1 if x+1<img.shape[0] else x):
            for j in range(y-1 if y-1>=0 else y, y+1 if y+1<img.shape[1] else y):
                img[i, j] = np.array([0, 0, 255])

    output_path = args.image_path[0: args.image_path.rfind('.')] + '_%d.png' % (args.threshold)
    cv2.imwrite(output_path, img)

if __name__ == '__main__':
    main()