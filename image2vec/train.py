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
        #x.rand()
        self.basis_vectors.append(x)

        masked_v = pyhdc.LBV()
        masked_v.xor(v)
        masked_v.xor(x)
        self.masked_vectors.append(masked_v)

    def build(self, to_adjust=-1):
        print ("\tmemory vectors:", self.vcount)
        if (to_adjust > 0): print ("\t\tadjusting to:", to_adjust)

        self.m.xor(self.m)
        if (self.vcount == 0):
            self.m.rand()
            return

        random_vectors = []
        if (to_adjust > self.vcount):
            x = pyhdc.LBV() 
            x.rand()
            random_vectors.append(x)

        csum_vectors = self.masked_vectors + random_vectors
        if (len(csum_vectors) == 1):
            self.m.xor(csum_vectors[0])
            self.masked_vectors = []
            return

        th = self.vcount // 2 
        for i in range(pyhdc.get_vector_width()):
            cnt = 0
            for v in csum_vectors:
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
                break
        return min_score, min_id


class Model:
    def __init__(self, m):
        self.cl_mapper = m
        self.bins = {}

    def add(self, vec_image, val):
        classes = self.cl_mapper.get_class(val)
        print ("\tassigning classes:", classes)
        for cl in classes:
            if cl not in self.bins.keys():
                self.bins[cl] = Memory()
            self.bins[cl].add(vec_image.vec)
    
    def build(self):
        print ("Bulding the model:", len(self.bins.keys()), "clusters")
        to_adjust = 0
        for i, cl in enumerate(sorted(self.bins.keys())):
            if (self.bins[cl].vcount > to_adjust): to_adjust = self.bins[cl].vcount
        if (to_adjust % 2 == 0): to_adjust += 1
        print ("Adjusting memory bins to:", to_adjust)
        for i, cl in enumerate(sorted(self.bins.keys())):
            print ("Building memory for cluster", cl, "(", i, "/", len(self.bins.keys()), ")")    
            self.bins[cl].build(-1)

    def infer(self, vec_image):
        clusters = []
        scores = []
        
        for i, cl in enumerate(sorted(self.bins.keys())):
            print ("\tLooking for a vector in memory", cl, "(", i, "/", len(self.bins.keys()), ")")
            score, id_ = self.bins[cl].find(vec_image.vec)
            print ("\tScore:", score, "\tbasis id:", id_)
  
            clusters.append(cl)
            scores.append(score)

        result = sorted(zip(scores, clusters))
        #print ("\tresults:", zip(scores, clusters))
        print ("\tcluster:", result[0][1])

        return self.cl_mapper.get_val_range([result[0][1]]), result[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default='.',
                        required=False)
    parser.add_argument('--width',
                        type=float,
                        required=False,
                        default=0.2)
    parser.add_argument('--mode',
                        type=int,
                        required=False,
                        default=0)

    args = parser.parse_args()

    print ("Opening", args.base_dir)

    TZ_mapper = ClassMapper(0.001, 0.001)
    TZModel = Model(TZ_mapper)

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

    vis_dir   = os.path.join(args.base_dir, 'vis')
    pydvs.replace_dir(vis_dir)

    #fake_images = []
    #for i, time in enumerate(gt_ts):
    #    x = pyhdc.LBV()
    #    x.rand()
    #    fake_images.append(x)


    #m = Memory()
    for i, time in enumerate(gt_ts):
        if (time > last_ts or time < first_ts):
            continue

        print ("Training:", i, "/", len(gt_ts))

        sl, _ = pydvs.get_slice(cloud, idx, time, args.width, args.mode, discretization)
        vec_image = VecImageCloud((180, 240), sl)
        
        #vec_image.x_err = gQ_y[i] * 10000
        #vec_image.y_err = gQ_y[i] * 10000
        #vec_image.z_err = gQ_y[i] * 10000
        #vec_image.vec = vec_image.image2vec()

        TZModel.add(vec_image, gQ_y[i])
        #m.add(vec_image.vec)
        #m.add(fake_images[i])

        if (i > 500):
            break

    TZModel.build()
    #m.build()

    print ("\n\n\n------------")
    
    x_axis = []
    gt_val = []
    
    hash_x = []
    hash_y = []
    hash_z = []
    hash_e = []
    hash_a = []

    lo_val = []
    hi_val = []

    for i, time in enumerate(gt_ts):
        if (time > last_ts or time < first_ts):
            continue
        sl, _ = pydvs.get_slice(cloud, idx, time, args.width, args.mode, discretization)
        vec_image = VecImageCloud((180, 240), sl)

        #vec_image.x_err = gQ_y[i] * 10000
        #vec_image.y_err = gQ_y[i] * 10000
        #vec_image.z_err = gQ_y[i] * 10000
        #vec_image.vec = vec_image.image2vec()

        print ("\nInference", i, ":")
        (lo, hi), [score, cl] = TZModel.infer(vec_image)

        x_axis.append(i)
        gt_val.append(gQ_y[i])
        
        hash_x.append(vec_image.x_err)
        hash_y.append(vec_image.y_err)
        hash_z.append(vec_image.z_err)
        hash_e.append(vec_image.e_count)
        hash_a.append(vec_image.nz_avg)

        lo_val.append(lo)
        hi_val.append(hi)

        #score, id_, lst = m.find(vec_image.vec)

        #cv2.imwrite(os.path.join(vis_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), vec_image.vis * 255)

        #score, id_ = m.find(fake_images[i])
        #print (i, ":", score, id_)

    fig, axs = plt.subplots(6, 1)

    axs[0].plot(x_axis, gt_val)
    axs[0].plot(x_axis, lo_val) 
    axs[0].plot(x_axis, hi_val)

    axs[1].plot(x_axis, hash_x)
    axs[2].plot(x_axis, hash_y)
    axs[3].plot(x_axis, hash_z)
    axs[4].plot(x_axis, hash_e)
    axs[5].plot(x_axis, hash_a)

    plt.show()
