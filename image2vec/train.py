#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, shutil, signal, glob, time
import matplotlib.pyplot as plt
from image2vec import *


class Memory:
    def __init__(self):
        self.basis_vectors = []
        self.masked_vectors = []
        self.vcount = 0
        self.m = pyhdc.LBV()
    
    def add(self, v):
        self.vcount += 1
        x = pyhdc.LBV() 
        x.rand()
        self.basis_vectors.append(x)

        masked_v = pyhdc.LBV()
        masked_v.xor(v)
        masked_v.xor(x)
        self.masked_vectors.append(masked_v)

    def build(self):
        print ("\tmemory vectors:", self.vcount)

        self.m.xor(self.m)
        if (self.vcount == 0):
            self.m.rand()
            return

        if (self.vcount == 1):
            self.m.xor(self.masked_vectors[0])
            self.masked_vectors = []
            return

        th = self.vcount // 2 
        for i in range(pyhdc.get_vector_width()):
            cnt = 0
            for v in self.masked_vectors:
                if (v.get_bit(i)):
                    cnt += 1
            if (cnt >= th):
                self.m.flip(i)
        self.masked_vectors = []

    def find(self, v):
        mem_test = pyhdc.LBV()
        mem_test.xor(self.m)
        mem_test.xor(v)

        min_score = pyhdc.get_vector_width()
        min_id = -1
        
        lst = []
        for i, b in enumerate(self.basis_vectors):
            tmp = pyhdc.LBV()
            tmp.xor(mem_test)
            tmp.xor(b)
            score = tmp.count()
            
            lst.append(score)
            if (min_score > score):
                min_score = score
                min_id = i
        return min_score, min_id, lst


class Model:
    def __init__(self, m):
        self.cl_mapper = m
        self.bins = {}

    def add(self, vec_image, val):
        classes = self.cl_mapper.get_class(val)
        for cl in classes:
            if cl not in self.bins.keys():
                self.bins[cl] = Memory
            self.bins[cl].add(vec_image.vec)
    
    def build(self):
        print ("Bulding the model:", len(self.bins.keys()), "clusters")
        for i, cl in enumerate(sorted(self.bins.keys())):
            print ("Building memory for cluster", cl, "(", i, "/", len(self.bins.keys()), ")")
            self.bins[cl].build()

    def infer(self, vec_image):
        clusters = []
        scores = []
        for i, cl in enumerate(sorted(self.bins.keys())):
            print ("Looking for a vector in memory", cl, "(", i, "/", len(self.bins.keys()), ")")
            score, id_ = self.bins[cl].find(vec_image.vec)
            print ("\tScore:", score, "\tbasis id:", id_)
            
            cluster.append(cl)
            scores.append(score)

        result = sorted(zip(scores, clusters))
        return self.cl_mapper.get_val_range([result[0][1]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default='.',
                        required=False)
    parser.add_argument('--width',
                        type=float,
                        required=False,
                        default=0.05)
    parser.add_argument('--mode',
                        type=int,
                        required=False,
                        default=0)

    args = parser.parse_args()

    print ("Opening", args.base_dir)

    TZ_mapper = ClassMapper(0.01, 0.005)
    TZModel = Model(TZ_mapper)

    #sys.exit()

    sl_npz = np.load(args.base_dir + '/recording.npz')
    cloud          = sl_npz['events']
    idx            = sl_npz['index']
    discretization = sl_npz['discretization']
    K              = sl_npz['K']
    D              = sl_npz['D']
    gt_poses       = sl_npz['poses']
    gt_ts          = sl_npz['gt_ts']
    gT_x           = sl_npz['Tx']
    gT_y           = sl_npz['Ty']
    gT_z           = sl_npz['Tz']
    gQ_x           = sl_npz['Qx']
    gQ_y           = sl_npz['Qy']
    gQ_z           = sl_npz['Qz']


    first_ts = cloud[0][0]
    last_ts = cloud[-1][0]


    fake_images = []
    for i, time in enumerate(gt_ts):
        x = pyhdc.LBV()
        x.rand()
        fake_images.append(x)


    m = Memory()
    for i, time in enumerate(gt_ts):
        if (time > last_ts or time < first_ts):
            continue

        print ("Training:", i, "/", len(gt_ts))

        sl, _ = pydvs.get_slice(cloud, idx, time, args.width, args.mode, discretization)
        #TZModel.add(VecImageCloud((180, 240), sl), gT_z[i])

        
        m.add(VecImageCloud((180, 240), sl).vec)
        
        #m.add(fake_images[i])

        if (i > 10):
            break

    m.build()
    
    print ("\n\n\n------------")
    for i, time in enumerate(gt_ts):
        if (time > last_ts or time < first_ts):
            continue


        sl, _ = pydvs.get_slice(cloud, idx, time, args.width, args.mode, discretization)
        score, id_, lst = m.find(VecImageCloud((180, 240), sl).vec)

        #score, id_ = m.find(fake_images[i])
        print (i, ":", score, id_, "|", lst)

