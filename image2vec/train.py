#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, shutil, signal, glob, time
from mpl_toolkits.mplot3d import Axes3D
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
        #print ("\tmemory vectors:", self.vcount)
        #if (to_adjust > 0): print ("\t\tadjusting to:", to_adjust)

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

        #print ("Outlier check...")
        #for v in self.masked_vectors:
        #    score, id_ = self.find(v)
        #    print ("\t", score, id_)

        print ("\n")
        self.masked_vectors = []

    def find(self, v):
        mem_test = pyhdc.LBV()
        mem_test.xor(self.m)
        mem_test.xor(v)

        min_score = pyhdc.get_vector_width()
        min_id = -1
        
        for i, b in enumerate(self.basis_vectors):
            tmp = pyhdc.LBV()
            tmp.xor(mem_test)
            tmp.xor(b)
            score = tmp.count()
            
            if (min_score > score):
                min_score = score
                min_id = i
                break
        return min_score, min_id


class Model:
    def __init__(self, m):
        self.cl_mapper = m
        self.bins = {}
        self.infer_db = []

    def add(self, vec_image, val):
        classes = self.cl_mapper.get_class(val)
        #print ("\tassigning classes:", classes)
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
            #print ("\tLooking for a vector in memory", cl, "(", i, "/", len(self.bins.keys()), ")")
            score, id_ = self.bins[cl].find(vec_image.vec)
            #print ("\tScore:", score, "\tbasis id:", id_)
  
            clusters.append(cl)
            scores.append(score)

        scores = np.array(scores, dtype=np.float)
        #scores /= float(pyhdc.get_vector_width())

        scores -= np.min(scores)
        scores /= np.max(scores)
        scores = 1 - scores

        #scores /= float(np.sum(scores))

        self.infer_db.append(scores)
        result = sorted(zip(scores, clusters))
        #print ("\tresults:", zip(scores, clusters))
        #print ("\tcluster:", result[-1][1])

        return self.cl_mapper.get_val_range([result[-1][1]]), result[-1]


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

    QY_mapper = ClassMapper(0.0001, 0.0001)
    QYModel = Model(QY_mapper)

    TX_mapper = ClassMapper(0.0001, 0.0001)
    TXModel = Model(TX_mapper)

    TZ_mapper = ClassMapper(0.0001, 0.0001)
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
 
    perf_cnt = 0.0
    im2vec_time = 0.0
    add_time = 0.0
    build_time = 0.0
    for i, t in enumerate(gt_ts):
        if (t > last_ts or t < first_ts):
            continue

        if (i % 100 == 0):
            print ("Training:", i, "/", len(gt_ts))

        sl, _ = pydvs.get_slice(cloud, idx, t, args.width, args.mode, discretization)
        
        start = time.time()
        vec_image = VecImageCloud((180, 240), sl)
        end = time.time()
        im2vec_time += end - start

        
        start = time.time()
        QYModel.add(vec_image, gQ_y[i])
        end = time.time()
        add_time += end - start

        TXModel.add(vec_image, gT_x[i])
        TZModel.add(vec_image, gT_z[i])

        perf_cnt += 1.0

        if (i > 3000):
            break

    start = time.time()
    QYModel.build()
    end = time.time()
    build_time = end - start

    TXModel.build()
    TZModel.build()

    print ("\nCount:", perf_cnt)
    print ("I2V time:", im2vec_time / perf_cnt)
    print ("ADD time:", add_time / perf_cnt)
    print ("Build time:", build_time)


    print ("\n\n\n------------")

    x_axis = []
    gt_tx = []
    gt_tz = []
    gt_qy = []

    hash_x = []
    hash_y = []
    hash_z = []
    hash_e = []
    hash_a = []
    hash_p = []
    hash_n = []
    hash_g = []

    lo_tx = []
    hi_tx = []
    lo_tz = []
    hi_tz = []
    lo_qy = []
    hi_qy = []

    perf_cnt = 0.0
    im2vec_time = 0.0
    infer_time = 0.0
    for i, t in enumerate(gt_ts):
        if (t > last_ts or t < first_ts):
            continue
        sl, _ = pydvs.get_slice(cloud, idx, t, args.width, args.mode, discretization)
        
        start = time.time()
        vec_image = VecImageCloud((180, 240), sl)
        end = time.time()
        im2vec_time += end - start

        if (i % 100 == 0):
            print ("Inference:", i, "/", len(gt_ts))
        
        start = time.time()
        (lo, hi), [score, cl] = QYModel.infer(vec_image)
        end = time.time()
        infer_time += end - start
        lo_qy.append(lo)
        hi_qy.append(hi)

        
        (lo, hi), [score, cl] = TXModel.infer(vec_image)
        lo_tx.append(lo)
        hi_tx.append(hi)
        
        (lo, hi), [score, cl] = TZModel.infer(vec_image)
        lo_tz.append(lo)
        hi_tz.append(hi)

        x_axis.append(i)
        gt_tx.append(gT_x[i])
        gt_tz.append(gT_z[i])
        gt_qy.append(gQ_y[i])
        
        #hash_x.append(vec_image.x_err)
        #hash_y.append(vec_image.y_err)
        #hash_z.append(vec_image.z_err)
        #hash_e.append(vec_image.e_count)
        #hash_a.append(vec_image.nz_avg)
        #hash_p.append(vec_image.p_count)
        #hash_n.append(vec_image.n_count)
        #hash_g.append(vec_image.g_count)

        perf_cnt += 1.0

        #score, id_, lst = m.find(vec_image.vec)

        #cv2.imwrite(os.path.join(vis_dir, 'frame_' + str(i).rjust(10, '0') + '.png'), vec_image.vis * 255)

        #score, id_ = m.find(fake_images[i])
        #print (i, ":", score, id_)

    print ("\nCount:", perf_cnt)
    print ("I2V time:", im2vec_time / perf_cnt)
    print ("Inference time:", infer_time / perf_cnt)


    print ("\n\n===============================")
    #print (len(TZModel.infer_db))
    #or l in TZModel.infer_db:
    #    print ("\t", len(l), l)

    #X = np.arange(0, len(QYModel.infer_db), 1)
    #Y = np.arange(0, len(QYModel.infer_db[0]), 1)
    #X, Y = np.meshgrid(X, Y)

    Z_tx = np.transpose(np.array(TXModel.infer_db))
    Z_tz = np.transpose(np.array(TZModel.infer_db))
    Z_qy = np.transpose(np.array(QYModel.infer_db))

    import scipy
    import scipy.ndimage

    gt_qy = scipy.ndimage.median_filter(gt_qy, 3)
    lo_qy = scipy.ndimage.median_filter(lo_qy, 21)
    hi_qy = scipy.ndimage.median_filter(hi_qy, 21)

    gt_tx = scipy.ndimage.median_filter(gt_tx, 3)
    lo_tx = scipy.ndimage.median_filter(lo_tx, 21)
    hi_tx = scipy.ndimage.median_filter(hi_tx, 21)

    gt_tz = scipy.ndimage.median_filter(gt_tz, 3)
    lo_tz = scipy.ndimage.median_filter(lo_tz, 21)
    hi_tz = scipy.ndimage.median_filter(hi_tz, 21)

    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(X, Y, Z, linewidth=0) 
    #fig.tight_layout()

    fig, axs = plt.subplots(6, 1)
    im0 = axs[0].imshow(Z_qy, origin='lower')
    axs[1].plot(x_axis, gt_qy)
    axs[1].plot(x_axis, lo_qy) 
    axs[1].plot(x_axis, hi_qy)

    im1 = axs[2].imshow(Z_tz, origin='lower')
    axs[3].plot(x_axis, gt_tz)
    axs[3].plot(x_axis, lo_tz) 
    axs[3].plot(x_axis, hi_tz)

    im2 = axs[4].imshow(Z_tx, origin='lower')
    axs[5].plot(x_axis, gt_tx)
    axs[5].plot(x_axis, lo_tx) 
    axs[5].plot(x_axis, hi_tx)

    #axs[2].plot(x_axis, hash_x)
    #axs[3].plot(x_axis, hash_y)
    #axs[4].plot(x_axis, hash_z)
    #axs[5].plot(x_axis, hash_e)
    #axs[6].plot(x_axis, hash_p)
    #axs[7].plot(x_axis, hash_n)
    #axs[8].plot(x_axis, hash_g)
    #axs[5].plot(x_axis, hash_a)

    plt.show()
