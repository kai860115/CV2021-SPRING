import numpy as np
import cv2
import argparse
import os
from matplotlib import pyplot as plt
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    rgb_setting = np.zeros((5,3))
    with open(args.setting_path) as f:
        f.readline()
        for i in range(5):
            rgb_setting[i] = np.array(f.readline().split(',')).astype(float)
        _, sigma_s, _, sigma_r = f.readline().split(',')
    sigma_s = int(sigma_s)
    sigma_r = float(sigma_r)

    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb)
    jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray)
    cost = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
    print("COLOR_BGR2GRAY cost = %f" % (cost))

    img_gray_h = img_gray
    cost_h = cost
    jbf_h = jbf_out
    img_gray_l = img_gray
    cost_l = cost
    jbf_l = jbf_out

    for r, g, b in rgb_setting:
        img_gray = r * img_rgb[:,:,2] + g * img_rgb[:,:,1] + b * img_rgb[:,:,0]
        img_gray = img_gray.astype(np.uint8)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray)
        cost = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
        if cost > cost_h:
            cost_h = cost
            img_gray_h = img_gray
            jbf_h = jbf_out
        if cost < cost_l:
            cost_l = cost
            img_gray_l = img_gray
            jbf_l = jbf_out
        print("(r,g,b) = (%.1f,%.1f,%.1f) cost = %f" % (r, g, b, cost))

    plt.figure(figsize=(16,8))
    plt.suptitle(args.image_path.split('/')[-1], fontsize=16)
    plt.subplot(231)
    plt.imshow(img_rgb)
    plt.title("Original RGB")
    plt.subplot(232)
    plt.imshow(jbf_h)
    plt.title("highest cost filter")
    plt.subplot(233)
    plt.imshow(jbf_l)
    plt.title("lowest cost filter")
    plt.subplot(234)
    plt.imshow(img_gray_h,cmap='gray')
    plt.title("highest cost grayscale")
    plt.subplot(235)
    plt.imshow(img_gray_l,cmap='gray')
    plt.title("lowest cost grayscale")
    output_path = args.image_path[0: args.image_path.rfind('.')] + '_result.png'
    plt.savefig(output_path)


if __name__ == '__main__':
    main()