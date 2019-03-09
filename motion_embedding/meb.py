#!/usr/bin/python3

import pyhdc
import argparse
import numpy as np
import os, sys, time

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2



def np_vec2c_vec(c_vec):
    x = pyhdc.LBV()
    if (not x.is_zero()):
        print ("ERROR! - starting with a nonzero vector")

    btc = 0
    for i, bit in enumerate(c_vec):
        if (bit == '1'):
            btc += 1
            x.flip(i)

    #if (pyhdc.get_vector_width() != len(c_vec)):
    #    print ("Vector length mismatch:", len(c_vec), pyhdc.get_vector_width())

    # check at least a bit count
    nbits = x.count()
    if (btc != nbits):
        print ("Bitcount mismatch!", nbits, btc)

    return x


def safe_hamming(v1, v2):
    x = pyhdc.LBV()
    x.xor(v1) # x = v1
    x.xor(v2) # x = v1 xor v2
    return x.count()


def get_X_y(base_dir, X, y, X_val, y_val, rate=10):
    with open(os.path.join(base_dir, 'im2vec.txt')) as fin:
        for i, line in enumerate(fin.readlines()):
            split_line = line.split(' ')
            vx = float(split_line[1])
            vy = float(split_line[2])
            vz = float(split_line[3])

            len2 = vx * vx + vy * vy + vz * vz
            if (len2 < 0.4):
                continue
            #l = sqrt(len2)

            vx /= 2.0
            vy /= 2.0
            vz /= 2.0

            # read the vector
            vec = np_vec2c_vec(split_line[0])

            if (i % rate == 0):
                X_val.append(vec)
                y_val.append([vx, vy, vz])
            else:
                X.append(vec)
                y.append([vx, vy, vz]) 


def color_scaled_square(img, scale, i, j):
    for k in range(i * scale, (i + 1) * scale):
        for l in range(j * scale, (j + 1) * scale):
            img[k,l] = 255;


def vec_visual(v, shape, scale=9):
    img = np.zeros((shape[0] * scale, shape[1] * scale), dtype=np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            idx = i * shape[0] + j
            if (v.get_bit(idx)):
                color_scaled_square(img, scale, i, j)
    return img


def vmap2images(vmap, scale=9):
    shape = (90, 90)


    for i, v in enumerate(vmap):
        if (i == 0) or (i > 254):
            continue
        
        img = vec_visual(v, shape, scale)
        name = "frame_" + str(i - 1).rjust(4, '0') + ".png"
        z = np.zeros((shape[0] * scale, shape[1] * scale), dtype=np.uint8)
        img = np.dstack((img, img, img))
        
        v_0 = pyhdc.LBV()
        v_0.xor(vmap[0])
        v_0.xor(v)
        v_0_img = vec_visual(v_0, shape, scale)

        img_r = np.copy(img)
        img_r[:,:,2] = v_0_img

        if (i > 0):
            v_prev_ = vmap[i - 1]
            v_prev = pyhdc.LBV()
            v_prev.xor(v_prev_)
            v_prev.xor(v)
            
            img_diff = vec_visual(v_prev, shape, scale)
            img[:,:,2] = img_diff

        
        sep = np.zeros((shape[0] * scale, 10, 3), dtype=np.uint8)

        img = np.hstack((img, sep, img_r))
        cv2.imwrite("/home/ncos/Desktop/vmap_viz/" + name, img)



class Vel2Vec:
    def __init__(self):
        self.vmap_paths = ['vmap_x.txt', 'vmap_y.txt', 'vmap_z.txt']
        self.vmaps = [self.read_vmap(path) for path in self.vmap_paths]
        vmap2images(self.vmaps[0])

    def vel2vecs(self, vel):
        ret = []

        nums = self.vel2nums(vel)
        for i, num in enumerate(nums):
            ret.append(self.num2vec(i, num))
        return ret

    def vel2nums(self, vel):
        ret = []
        for i, component in enumerate(vel):
            vi = component + 1.0
            # normalize
            if (vi < 0.0): vi = 0.0
            if (vi > 2.0): vi = 2.0
            vi = int(vi / 2 * 254)
            if (vi > 254): vi = 254
            ret.append(vi)
        return ret

    def num2vec(self, i, num):
        base_vec = pyhdc.LBV()
        base_vec.xor(self.vmaps[i][num])
        return base_vec

    def read_vmap(self, path):
        print ("\n=======================\nReading velocity vmap...")
        vmap = []
        f = open(path, 'r')
        for i, line in enumerate(f.readlines()):
            v = np_vec2c_vec(line)
            vmap.append(v)
        f.close()

        print ("Read", len(vmap), "vectors")
        print ()
        return vmap


def csum(vectors):
    th = len(vectors) // 2 
    ret = pyhdc.LBV()
    for i in range(pyhdc.get_vector_width()):
        cnt = 0
        for v in vectors:
            if (v.get_bit(i)):
                cnt += 1
        if (cnt >= th):
            ret.flip(i)
    return ret


def bind(vectors, basis_vectors):
    scaled_v = []
    for i, v in enumerate(vectors):
        x = pyhdc.LBV()
        x.xor(v)
        x.xor(basis_vectors[i])
        scaled_v.append(x)

    return csum(scaled_v)


def memscore(v_img, M, M_img):
    for i in range(len(M)):
        h0 = safe_hamming(M[i], v_img) # should be 4k
        h1 = safe_hamming(M_img[i], v_img) # should become small
        count_raw = M[i].count()
        count_ub  = M_img[i].count()
        
        print (i, ": ", h0, "->", h1, "|", count_raw, "->", count_ub)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--big', action='store_true')
    #parser.add_argument('--batch',
    #                    type=int,
    #                    default=32,
    #                    required=False)
    args = parser.parse_args()

    X = []
    y = []
    X_val = []
    y_val = []

    get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_0", X, y, X_val, y_val)
    #get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_1", X, y, X_val, y_val)
    #get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_3", X, y, X_val, y_val)
    
    if (args.big):
        print ("Processing big dataset")
        get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_2", X, y, X_val, y_val)
        get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_FAST", X, y, X_val, y_val)
        get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_PLAIN_WALL", X, y, X_val, y_val)
        get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_PLAIN_WALL_II", X, y, X_val, y_val)
        get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_S3", X, y, X_val, y_val)
        get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O3_PLAIN_WALL_P3", X, y, X_val, y_val)
        #get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O3_TOP", X, y, X_val, y_val)

    print ("Input dataset size:", len(X), "data points")
    print ("\t\t", len(X_val), "validation points")
    print ()

    vel_converter = Vel2Vec()
    MEMORY = []
    MEMORY_image = []

    basis_vectors = []
    for i in range(4):
        x = pyhdc.LBV()
        for k in range(i + 1):
            x.rand()
        basis_vectors.append(x)

    # create memory
    for i, v_img in enumerate(X):
        v_vel = vel_converter.vel2vecs(y[i])
        v_mem = bind([v_img] + v_vel, basis_vectors)
        MEMORY.append(v_mem)

        test = pyhdc.LBV()
        test.xor(v_mem)
        test.xor(basis_vectors[0])

        MEMORY_image.append(test)


        #h0 = safe_hamming(v_mem, v_img)
        #h1 = safe_hamming(test, v_img)
        #count_raw = v_mem.count()
        #count_ub  = test.count()
        #print (i, ": ", h0, "->", h1, "|", count_raw, "->", count_ub)



    memscore(X[100], MEMORY, MEMORY_image)



        #if (i % 1000):
        #    print ("Adding to memory:", i, "/", len(X))

    # evaluating on X
    #for v_img, i in enumerate(X):
    #    scores = []
        
    #    for M in MEMORY_image:
            
            
