#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, signal, math, time
import matplotlib.colors as colors

import pydvs, pyhdc, cv2


def colorizeImage(flow_x, flow_y):
    hsv_buffer = np.empty((flow_x.shape[0], flow_x.shape[1], 3))
    hsv_buffer[:,:,1] = 1.0
    hsv_buffer[:,:,0] = (np.arctan2(flow_y, flow_x) + np.pi)/(2.0 * np.pi)
    hsv_buffer[:,:,2] = np.linalg.norm( np.stack((flow_x, flow_y), axis=0), axis=0 )
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


class VecImageRaw:
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
        I = VecImageRaw(os.path.join(args.f1, img_path))
        images_f1.append(I)
    return images_f1

# =================================

class VecImageCloud:
    def __init__(self, shape, cloud):
        self.shape = shape
        self.width = 0
        if (cloud.shape[0] > 0):
            self.width = cloud[-1][0] - cloud[0][0]

        # Compute images according to the model
        dvs_img = pydvs.dvs_img(cloud, self.shape, model=[0, 0, 0, 0], 
                                scale=1, K=None, D=None)
        #dvs_img = np.copy(dvs_img[:50,:50,:])
        
        # Compute errors on the images
        dgrad = np.zeros((dvs_img.shape[0], dvs_img.shape[1], 2), dtype=np.float32)
        self.x_err, self.y_err, self.yaw_err, self.z_err, self.e_count, self.nz_avg = \
            pydvs.dvs_flow_err(dvs_img, dgrad)

        print (self.x_err, self.y_err, self.z_err, self.yaw_err, self.e_count, self.nz_avg)

        # Visualization
        c_img = dvs_img[:,:,0] + dvs_img[:,:,2]
        c_img = np.dstack((c_img, c_img, c_img)) * 0.5 / (self.nz_avg + 1e-3)

        dvs_img[:,:,1] *= 1.0 / self.width
        t_img = np.dstack((dvs_img[:,:,1], dvs_img[:,:,1], dvs_img[:,:,1]))
        G_img = colorizeImage(dgrad[:,:,0], dgrad[:,:,1])

        self.vec = self.image2vec(dgrad)

        cv2.namedWindow('GUI', cv2.WINDOW_NORMAL)
        cv2.imshow('GUI', np.hstack((c_img, t_img, G_img)))
        cv2.waitKey(0) 


    def num2vec(self, num, size=500):
        n = int(num)
        min_ = -size // 2
        n_bits = n - min_
        if (n_bits < 0): n_bits = 0
        if (n_bits > size): n_bits = size
        return n_bits


    def image2vec(self, dgrad):
        ret = pyhdc.LBV()
        params = [self.x_err, self.y_err, self.z_err]

        chunk_size = 500
        to_encode = [self.num2vec(p, chunk_size) for p in params]
        
        for i, n_bits in enumerate(to_encode):
            start_offset = i * chunk_size
            for j in range(n_bits):
                ret.flip(start_offset + j)

        return ret


def crossEvaluateHamming(images, rjust=5):
    ret = ""
    for i in range(len(images)):
        for j in range (len(images)):
            if (j < i):
                ret += "-".rjust(rjust) + " "
                continue
            score = safeHamming(images[i].vec, images[j].vec)
            ret += str(score).rjust(rjust) + " "
        ret += "\n"
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('f1',
                        type=str)
    parser.add_argument('f2',
                        type=str)
    parser.add_argument('--width',
                        type=float,
                        required=False,
                        default=0.2)
    args = parser.parse_args()
 
    sl_npz_1 = np.load(args.f1)
    cloud_1          = sl_npz_1['events']
    idx_1            = sl_npz_1['index']
    discretization_1 = sl_npz_1['discretization']
    length_1 = cloud_1[-1][0] - cloud_1[0][0]
    n_1 = 6
    print ("Length 1:", length_1)

    images_f1 = []
    for i in range(n_1):
        sl, _ = pydvs.get_slice(cloud_1, idx_1, i * length_1 / n_1, args.width, 0, discretization_1)
        images_f1.append(VecImageCloud((180, 240), sl))

    # Evaluate:
    print(crossEvaluateHamming(images_f1))



    #sl_npz_2 = np.load(args.f2)
    #cloud_2          = sl_npz_2['events']
    #idx_2            = sl_npz_2['index']
    #discretization_2 = sl_npz_2['discretization']

    #images_f1 = getImageFolder(args.f1)
    #images_f2 = getImageFolder(args.f2)
