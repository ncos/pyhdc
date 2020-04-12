#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, signal, math, time
import matplotlib.colors as colors
import random

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


class VecImageCloud:
    def __init__(self, shape, cloud, direct=False):
        random.seed()
        self.shape = shape
        self.width = 0
        if (cloud.shape[0] > 0):
            self.width = cloud[-1][0] - cloud[0][0]

        # Compute images according to the model
        self.dvs_img = pydvs.dvs_img(cloud, self.shape, model=[0, 0, 0, 0],
                                     scale=1, K=None, D=None)
        #self.dvs_img[:,:,1] = cv2.medianBlur(self.dvs_img[:,:,1], 5)
        self.dvs_img[:,:,1] = cv2.GaussianBlur(self.dvs_img[:,:,1], (7,7), 0)

        #dvs_img = np.copy(dvs_img[:50,:50,:])

        # Compute errors on the images
        dgrad = np.zeros((self.dvs_img.shape[0], self.dvs_img.shape[1], 2), dtype=np.float32)
        self.x_err2, self.y_err2, self.yaw_err, self.z_err2, self.e_count, self.nz_avg = \
            pydvs.dvs_flow_err(self.dvs_img, dgrad)

        dgrad_ = np.zeros((self.dvs_img.shape[0], self.dvs_img.shape[1], 2), dtype=np.float32)
        self.x_err, self.y_err, self.yaw_err, self.z_err, self.e_count, self.nz_avg = \
            pydvs.dvs_err(self.dvs_img, dgrad_)

        self.x_err *= 10
        self.y_err *= 10
        self.z_err /= 5

        self.x_err2 /= 5
        self.y_err2 /= 5
        self.z_err2 /= 500
        self.e_count /= 6250
        self.e_count -= 1
        self.nz_avg /= 4
        self.nz_avg -= 1

        self.p_count = np.sum(self.dvs_img[:,:,0]) / 20000
        self.n_count = np.sum(self.dvs_img[:,:,2]) / 30000
        self.g_count = self.p_count + self.n_count / 50000
        self.p_count -= 1
        self.n_count -= 1
        self.g_count -= 1

        if (direct):
            self.vec = self.image2vec_direct(dgrad)
        else:
            self.vec = self.image2vec(dgrad)

        return

        #print (self.x_err, self.y_err, self.z_err, self.yaw_err, self.e_count, self.nz_avg)

        # Visualization
        c_img = self.dvs_img[:,:,0] + self.dvs_img[:,:,2]
        c_img = np.dstack((c_img, c_img, c_img)) * 0.5 / (self.nz_avg + 1e-3)

        self.dvs_img[:,:,1] *= 1.0 / self.width
        t_img = np.dstack((self.dvs_img[:,:,1], self.dvs_img[:,:,1], self.dvs_img[:,:,1]))
        G_img = colorizeImage(dgrad[:,:,0], dgrad[:,:,1])

        self.vis = np.hstack((t_img, G_img))

        cv2.namedWindow('GUI', cv2.WINDOW_NORMAL)
        cv2.imshow('GUI', np.hstack((t_img, G_img)))
        cv2.waitKey(0)


    def num2vec(self, num, size):
        if (np.isnan(num)): return np.nan
        n = int(num * size / 2)
        min_ = -size // 2
        n_bits = n - min_
        if (n_bits < 0): n_bits = 0
        if (n_bits > size): n_bits = size
        return n_bits


    def image2vec(self, dgrad=None):
        ret = pyhdc.LBV()
        #params = [self.z_err / 2]
        #params = [self.x_err2, self.y_err2, self.x_err, self.y_err]
        params = [self.x_err2, self.y_err2, self.x_err, self.y_err, self.z_err, self.z_err2]
        #params = [self.x_err2, self.y_err2, self.x_err, self.y_err, self.z_err, self.z_err2, self.e_count]
        #params = [self.x_err2, self.y_err2, self.z_err2]
        #params = [self.x_err, self.y_err, self.z_err, self.g_count]
        #params = [self.x_err, self.y_err, self.z_err, self.x_err, self.y_err, self.z_err, self.e_count, self.p_count, self.n_count, self.g_count]

        #step = 50
        #for i in range(self.dvs_img.shape[0] // step):
        #    for j in range(self.dvs_img.shape[1] // step):
        #        dvs_img_ = np.copy(self.dvs_img[i:i+step,j:j+step,:])
        #        dgrad_ = np.zeros((dvs_img_.shape[0], dvs_img_.shape[1], 2), dtype=np.float32)
        #        x_err, y_err, yaw_err, z_err, e_count, nz_avg = \
        #                pydvs.dvs_flow_err(dvs_img_, dgrad_)
        #
        #        if (e_count < 20):
        #            x_err = np.nan
        #            y_err = np.nan
        #            z_err = np.nan

        #        e_count /= (6250 / (self.dvs_img.shape[0] * self.dvs_img.shape[1])) * step * step
        #        e_count -= 1

                #params.append(x_err / 5)
                #params.append(y_err / 5)
                #params.append(z_err / 500)
                #params.append(e_count)

        #print (self.dvs_img.shape)
        #print (params)
        #print (len(params))

        chunk_size = 500
        to_encode = [self.num2vec(p, chunk_size) for p in params]
        scale = 2

        #ret.rand()
        for i, n_bits in enumerate(to_encode):
            start_offset = i * chunk_size * scale
            end_chunk = (i + 1) * chunk_size * scale
            if (np.isnan(params[i])):
                continue

            #for j in range(start_offset, n_bits + start_offset):
            #    ret.flip(j)

            for j in range(start_offset, end_chunk):
                if (j <= n_bits + start_offset and not ret.get_bit(j)):
                    ret.flip(j)
                if (j > n_bits + start_offset and ret.get_bit(j)):
                    ret.flip(j)

        return ret


    def image2vec_direct(self, dgrad):
        # convert every pixel tp hbv
        chunk_size = 500

        # encode every individual pixel value with a vector
        vectors = []
        for i in range(dgrad.shape[0]):
            for j in range(dgrad.shape[1]):
                if (dgrad[i][j][0] == 0 or dgrad[i][j][1] == 0): continue

                # encode first pixel value
                tmp0 = pyhdc.LBV()
                nbts0 = self.num2vec(dgrad[i][j][0] / 5, chunk_size)
                for k in range(nbts0):
                    tmp0.flip(k)
                tmp0.permute('x', 0)

                # encode second  pixel value
                tmp1 = pyhdc.LBV()
                nbts1 = self.num2vec(dgrad[i][j][1] / 5, chunk_size)
                for k in range(nbts1):
                    tmp1.flip(k)
                tmp1.permute('y', 0)

                # v = XOR(P_1(v0), P_2(v1))
                tmp = pyhdc.LBV()
                vectors.append(tmp)
                vectors[-1].xor(tmp0)
                vectors[-1].xor(tmp1)


        # apply consensus sum on all vectors
        ret = pyhdc.csum(vectors, '1')

        return ret


class ClassMapper:
    def __init__(self, bsize_, stride_):
        self.bsize = float(bsize_)
        self.stride = float(stride_)

    def get_class(self, val):
        v = float(val)
        cl0 = int(v / self.stride)
        classes = [cl0]
        i = 1
        while (v < ((cl0 - i) * self.stride + self.bsize)):
            classes.append(cl0 - i)
            i += 1
        return sorted(classes)

    def get_val_range(self, classes):
        c = sorted(classes)
        return [c[-1] * self.stride, c[0] * self.stride + self.bsize]
