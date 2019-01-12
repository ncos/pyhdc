#!/usr/bin/python3

import pyhdc
import sys, os

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
            v.xor(vmap[val])

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
            x.xor(vmap[val])
        
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


if __name__ == '__main__':
    print ("Max order on x:", pyhdc.get_max_order('x'))
    print ("Max order on y:", pyhdc.get_max_order('y'))
    print ("Vector width:", pyhdc.get_vector_width(), "bits")
    print ()

    # load the dataset
    X = []
    y = []
    get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_3", X, y)

    # vector map
    vmap = []
    for i in range(256):
        v = pyhdc.LBV()
        v.rand()
        vmap.append(v)

    # convert to vectors
    print ("Processing", len(X), "images")
    vectors = [img2vec(cv2.imread(img_name, cv2.IMREAD_UNCHANGED), vmap) for img_name in X]

