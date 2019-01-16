#!/usr/bin/python3

import pyhdc
import argparse
import numpy as np
import os, sys, time


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





def csum(vectors):
    ret = pyhdc.LBV()
    if (len(vectors) == 1):
        ret.xor(vectors[0])
        return ret

    th = len(vectors) // 2 
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


def memscore(v_img, M, M_img, seq_len, basis_vectors):
    res = []
    for i in range(len(M)):
        h0 = safe_hamming(M[i], v_img) # should be 4k
        h1 = safe_hamming(M_img[i], v_img) # should become small
        count_raw = M[i].count()
        count_ub  = M_img[i].count()
       
        test = pyhdc.LBV()
        test.xor(M[i])
        test.xor(v_img)

        res.append(h1)

        #s = str(i).rjust(4) + str(h1).rjust(5) + "|"
        #for i in range(0, seq_len):
        #    h = safe_hamming(test, basis_vectors[i]) # should become small?
        #    s += str(h).rjust(5)

        #print (s)
        #print (i, ": ", h1, "|")
        #print (i, ": ", h0, "->", h1, "|", count_raw, "->", count_ub)
    return min(res)


def create_memory(X, seq_len, basis_vectors):
    MEMORY = []
    MEMORY_image = []

    for i in range(0, len(X) - seq_len, seq_len):
        vecs = []
        for j in range(i, i + seq_len):
            vecs.append(X[j])
        
        v_mem = bind(vecs, basis_vectors)
        MEMORY.append(v_mem)

        test = pyhdc.LBV()
        test.xor(v_mem)
        test.xor(basis_vectors[0])

        MEMORY_image.append(test)
        break

    return MEMORY, MEMORY_image



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


    max_bb = 900
    X = X[100:]
    #X = X[100:200 + max_bb]


    print ("Input dataset size:", len(X), "data points")
    print ("\t\t", len(X_val), "validation points")
    print ()


    basis_vectors = []
    for i in range(max_bb):
        x = pyhdc.LBV()
        for k in range(i + 1):
            x.rand()
        basis_vectors.append(x)

    seq_len = 3

    print ("Creating a memory")
    
    #MEMORY, MEMORY_image = create_memory(X, seq_len, basis_vectors)
    #h = memscore(X[0], MEMORY, MEMORY_image, seq_len, basis_vectors)
    #print (h)
    
    #sys.exit()
    for seq_len in range(1, 10000):
        MEMORY, MEMORY_image = create_memory(X, seq_len, basis_vectors)

        h = memscore(X[0], MEMORY, MEMORY_image, seq_len, basis_vectors)
        print (h)
