#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, random

import pyhdc


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
        for j in range(l):
            h = safe_hamming(vmap[i], vmap[j])
            s += str(h).rjust(5)
        s += "\n"

    f.write(s)
    f.close()
    print ()


def permute(vmap, num=50):
    print ("Random vector permutation...")

    axes = ['x', 'y']
    permutations = [['x', 0], ['y', 0]]
    for i in range(num):
        axis = random.choice(axes)
        nperm = random.randint(0, pyhdc.get_max_order(axis))
        permutations.append([axis, nperm])

    for v in vmap:
        for p in permutations:
            v.inv_permute(p[0], p[1])

    print ("Done.")
    print ()


def add_noise(vmap, upto=10):
    print ("Add noise...")

    v_len = pyhdc.get_vector_width()
    for v in vmap:
        times = random.randint(0, upto)
        for i in range(times):
            bit = random.randint(0, v_len - 1)
            v.flip(bit)

    print ("Done.")
    print ()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size',
                        type=int,
                        default=25,
                        required=False)
    parser.add_argument('--fill_rate',
                        type=float,
                        default=0.5,
                        required=False)
    parser.add_argument('--noise',
                        type=int,
                        default=10,
                        required=False)
    parser.add_argument('-o',
                        type=str,
                        default="vmap.txt",
                        required=False)
    args = parser.parse_args()

    v_len = pyhdc.get_vector_width()

    print ("Max order on x:", pyhdc.get_max_order('x'))
    print ("Max order on y:", pyhdc.get_max_order('y'))
    print ("Vector width:", pyhdc.get_vector_width(), "bits")
    print ()

    patch_size = int(args.fill_rate * v_len)
    stride = int(v_len / (4 * args.size))
    z_fill = int(v_len / 4)

    print ("patch_size:", patch_size)
    print ("stride:", stride, "; hamming =", 2 * stride)
    print ("z-fill:", z_fill)
    print ()

    vmap = []
    for i in range(args.size):
        v = pyhdc.LBV()
        start_bit = stride * i
        end_bit = min(start_bit + patch_size, v_len - 1)
        for bit_idx in range(start_bit, end_bit + 1):
            v.flip(bit_idx)
        vmap.append(v)
    # 'gravity' vector
    v = pyhdc.LBV()
    for i in range(0, v_len - stride, 2 * stride):
        for bit_idx in range(i, i + stride):
            v.flip(bit_idx)
    vmap.append(v)
    
    # permute the whole thing
    permute(vmap, 50)
    
    # add random noise
    add_noise(vmap, args.noise)

    print ("Gravity count:", v.count())
    check_vmap(vmap)

    f = open(args.o, 'w')
    for i, v in enumerate(vmap):
        s = v.__repr__()
        f.write(s + "\n")

    f.close()
