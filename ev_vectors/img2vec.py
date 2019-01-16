#!/usr/bin/python3

import pyhdc
import argparse
import numpy as np
import os, sys, shutil, signal, glob, time

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2



def get_X_y(base_dir, X, y):
    with open(os.path.join(base_dir, 'cam_vels_local_frame.txt')) as fin:
        for line in fin.readlines():
            split_line = line.split(' ')
            img_path = os.path.join(base_dir, 'slices', split_line[0])
            vx = float(split_line[1])
            vy = float(split_line[2])
            vz = float(split_line[3])

            X.append(img_path)
            y.append([vx, vy, vz])

def i2v_baseline(image, vmap):
    pixel_count = 0
    x = pyhdc.LBV()
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            val = image[i, j]
            if (val < 1):
                continue

            pixel_count += 1

            # copy the vector from map
            v = pyhdc.LBV()
            v.xor(vmap[val - 1])

            # permute
            if (i > 0):
                v.permute('x', i - 1)
            
            if (j > 0):
                v.permute('y', j - 1)
            
            # add to the accumulator
            x.xor(v)
    return x, pixel_count

def i2v_fast(image, vmap):
    pixel_count = 0
    x = pyhdc.LBV()
    for i in range(0, image.shape[0]):
        # permute
        if (i > 0):
            x.permute('x', i - 1)

        for j in range(0, image.shape[1]):
            val = image[i, j]
            if (val < 1):
                continue

            pixel_count += 1

            # permute
            if (j > 0):
                x.permute('y', j - 1)

            # add to the accumulator
            x.xor(vmap[val - 1])
        
    return x, pixel_count


def img2vec(img, vmap, use_fast=False):
    # Make sure it is the grayscale image
    image = img
    if (len(img.shape) > 2 and img.shape[2] == 3):
        image = img[:,:,1]
    elif (len(img.shape) == 2):
        image = img
    else:
        print ("Unsupported image size: ", image.shape)
    if (len(vmap) != 256):
        print ("vmap should have length of 256, not", len(vmap))

    x = None
    pixel_count = 0
    if (use_fast):
        x, pixel_count = i2v_fast(image, vmap)
    else:
        x, pixel_count = i2v_baseline(image, vmap)

    fill = int(pixel_count / (image.shape[0] * image.shape[1]) * 100)
    print ("Image shape:", image.shape, "Type:", image.dtype, "Fill:", fill, "Count:", x.count())
    return x


def np_vec2c_vec(c_vec):
    x = pyhdc.LBV()
    if (x.is_zero()):
        print ("Starting conversion with a zero vector")
    else:
        print ("ERROR! - starting with a nonzero vector")

    btc = 0
    for i, bit in enumerate(c_vec):
        if (bit == '1'):
            btc += 1
            x.flip(i)

    if (pyhdc.get_vector_width() != len(c_vec)):
        print ("Vector length mismatch:", len(c_vec), pyhdc.get_vector_width())

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


def check_vmap(vmap):
    f = open('./vmap_check.txt', 'w')
    l = len(vmap)
    f.write("Vmap size: " + str(l) + "\n")
    print ("Checking vmap")
    s = ""
    for i in range(l):
        print ("Processing vmap vector", i + 1, "out of", l)
        for j in range(l):
            h = safe_hamming(vmap[i], vmap[j])
            s += str(h).rjust(5)
        s += "\n"

    f.write(s)
    f.close()
    print ()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default="",
                        required=True)
    args = parser.parse_args()

    print ("Max order on x:", pyhdc.get_max_order('x'))
    print ("Max order on y:", pyhdc.get_max_order('y'))
    print ("Vector width:", pyhdc.get_vector_width(), "bits")
    print ()

    # load the dataset
    X = []
    y = []
    get_X_y(args.base_dir, X, y)

    #X = X[:10]

    # vector map
    print ("\n=======================\nReading vmap...")
    vmap = []
    f = open('./vmap.txt', 'r')
    for i, line in enumerate(f.readlines()):
        v = np_vec2c_vec(line)
        #v = pyhdc.LBV()
        #v.rand()
        vmap.append(v)
    f.close()

    print ("Read", len(vmap), "vectors")
    print ()

    # sanity check
    check_vmap(vmap)

    # convert to vectors
    print ("Processing", len(X), "images")

    f = open(os.path.join(args.base_dir, 'im2vec.txt'), 'w')
    ref_vec = pyhdc.LBV()
    for i, img_name in enumerate(X):
        v = img2vec(cv2.imread(img_name, cv2.IMREAD_UNCHANGED), vmap)

        s = v.__repr__()
        for vel in y[i]:
            s += " " + str(vel)
        f.write(s + "\n")

        # ======
        if (i > 0):
            ref_vec.xor(v)
            hamming = ref_vec.count()
            print ("\t\tHamming = ", hamming)

        ref_vec.xor(ref_vec)
        ref_vec.xor(v)

    f.close()
