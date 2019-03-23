#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, signal, math, time
import matplotlib.colors as colors

import pydvs, cv2


def colorizeImage(flow_x, flow_y):
    hsv_buffer = np.empty((flow_x.shape[0], flow_x.shape[1], 3))
    hsv_buffer[:,:,1] = 1.0
    hsv_buffer[:,:,0] = (np.arctan2(flow_y, flow_x) + np.pi)/(2.0*np.pi)
    hsv_buffer[:,:,2] = np.linalg.norm( np.stack((flow_x,flow_y), axis=0), axis=0 )
    hsv_buffer[:,:,2] = np.log(1. + hsv_buffer[:,:,2])

    flat = hsv_buffer[:,:,2].reshape((-1))
    m = 1
    try:
        m = np.nanmax(flat[np.isfinite(flat)])
    except:
        m = 1
    if not np.isclose(m, 0.0):
        hsv_buffer[:,:,2] /= m

    return colors.hsv_to_rgb(hsv_buffer)


def safeHamming(v1, v2):
    x = pyhdc.LBV()
    x.xor(v1) # x = v1
    x.xor(v2) # x = v1 xor v2
    return x.count()


class VecImage:
    def __init__(self, path, ch=0):
        self.img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        self.channel = ch
        self.timg = self.img[:,:,self.channel].astype(np.float32) / 255.0
        self.cimg = self.img[:,:,(self.channel + 1)%3] + self.img[:,:,(self.channel + 2)%3]
        ret, self.mask = cv2.threshold(self.cimg, 20, 1, cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.float32)
        self.mask = cv2.erode(self.mask, kernel, iterations = 1)

        sx, sy = self.image2flow(self.img[:,:,self.channel])
        sx *= self.mask
        sy *= self.mask

        vis = colorizeImage(sx, sy)
        vis = np.hstack((vis, np.dstack((self.cimg, self.cimg, self.cimg)) / 255.0))
        vis = np.hstack((vis, np.dstack((self.mask, self.mask, self.mask))))

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', vis)
        cv2.waitKey(0) 


    def image2flow(self, img):
        sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        return sobelx, sobely


def getImageFolder(path):
    images_f1 = []
    for img_path in os.listdir(args.f1):
        I = VecImage(os.path.join(args.f1, img_path))
        images_f1.append(I)
    return images_f1


def crossEvaluateHamming(images, rjust=5):
    ret = ""
    for i in range(len(images)):
        for j in range (len(images)):
            if (j < i):
                ret += "-".rjust(rjust) + " "
                continue
            score = safeHamming(images[i].vec, images[j].vec)
            ret += str(score).rjust(rjust) + " "
        str += "\n"
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('f1',
                        type=str)
    parser.add_argument('f2',
                        type=str)

    args = parser.parse_args()

    images_f1 = getImageFolder(args.f1)
    #images_f2 = getImageFolder(args.f2)

    # Evaluate:
    print(crossEvaluateHamming(images_f1))
