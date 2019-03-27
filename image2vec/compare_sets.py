#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, signal, math, time
import matplotlib.colors as colors

from image2vec import *


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
