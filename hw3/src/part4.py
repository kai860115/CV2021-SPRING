import numpy as np
import cv2
import random
from numpy.core.fromnumeric import size
from numpy.linalg.linalg import solve
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)
np.random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(im1,None)
        kp2, des2 = orb.detectAndCompute(im2,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = list(filter(lambda x: x.distance < 60, matches))
        v_list = np.array([kp1[match.queryIdx].pt for match in matches])
        u_list = np.array([kp2[match.trainIdx].pt for match in matches])

        # TODO: 2. apply RANSAC to choose best H
        iterations = 25000
        threholds = 3
        best_H = np.eye(3)
        best_v_list = np.zeros(0)
        best_u_list = np.zeros(0)

        for _ in tqdm(range(iterations)):
            idx = np.random.choice(len(v_list), size=(4,), replace=False)
            v_sample = v_list[idx]
            u_sample = u_list[idx]
            H = solve_homography(u_sample, v_sample)
            v_list_estimate = H @ np.vstack((u_list[:, 0], u_list[:, 1], np.ones(len(u_list))))
            v_list_estimate = v_list_estimate[0:2] / v_list_estimate[2]
            v_list_estimate = v_list_estimate.T
            distances = np.linalg.norm(v_list - v_list_estimate, axis=1)
            if (distances < threholds).sum() >= len(best_v_list):
                best_v_list = v_list[distances < threholds]
                best_u_list = u_list[distances < threholds]
                best_H = H

            if (distances < threholds).sum() >= 0.95 * len(matches):
                break
        
        H = solve_homography(best_u_list, best_v_list)
        v_list_estimate = H @ np.vstack((u_list[:, 0], u_list[:, 1], np.ones(len(u_list))))
        v_list_estimate = v_list_estimate[0:2] / v_list_estimate[2]
        v_list_estimate = v_list_estimate.T
        distances = np.linalg.norm(v_list - v_list_estimate, axis=1)
        if (distances < threholds).sum() >= len(best_v_list):
            best_v_list = v_list[distances < threholds]
            best_u_list = u_list[distances < threholds]
            best_H = H

        # TODO: 3. chain the homographies
        last_best_H = last_best_H @ best_H

        # TODO: 4. apply warping
        out = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')

    return out


if __name__ == "__main__":

    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)