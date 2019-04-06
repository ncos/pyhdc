#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, shutil, signal, glob, time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import scipy
import scipy.ndimage

from image2vec import *
from model_a import *
from model_b import *


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

    QY_mapper = ClassMapper(0.0005, 0.0005)
    QYModel = Model_a(QY_mapper)

    TX_mapper = ClassMapper(0.002, 0.002)
    TXModel = Model_a(TX_mapper)

    TZ_mapper = ClassMapper(0.002, 0.002)
    TZModel = Model_a(TZ_mapper)


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
 

    gQ_y = scipy.ndimage.median_filter(gQ_y, 11)
    gT_x = scipy.ndimage.median_filter(gT_x, 11)
    gT_z = scipy.ndimage.median_filter(gT_z, 11)

    perf_cnt = 0.0
    im2vec_time = 0.0
    add_time = 0.0
    build_time = 0.0
    for i, t in enumerate(gt_ts):
        if (t > last_ts or t < first_ts):
            continue
        
        if (i > 2000): break
        #if (i > len(gt_ts) / 5 and i < 4 * len(gt_ts) / 5): continue


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

        #if (i > 250):
        #    break

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
        
        hash_x.append(vec_image.x_err)
        hash_y.append(vec_image.y_err)
        hash_z.append(vec_image.z_err)
        #hash_e.append(vec_image.e_count)
        #hash_a.append(vec_image.nz_avg)
        #hash_p.append(vec_image.p_count)
        #hash_n.append(vec_image.n_count)
        hash_g.append(vec_image.g_count)

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


    gt_qy = np.array(gt_qy) * 40
    #gt_qy = scipy.ndimage.median_filter(gt_qy, 3)
    lo_qy = scipy.ndimage.median_filter(lo_qy, 21) * 40
    hi_qy = scipy.ndimage.median_filter(hi_qy, 21) * 40

    gt_tx = np.array(gt_tx) * 40
    #gt_tx = scipy.ndimage.median_filter(gt_tx, 3)
    lo_tx = scipy.ndimage.median_filter(lo_tx, 21) * 40
    hi_tx = scipy.ndimage.median_filter(hi_tx, 21) * 40

    gt_tz = np.array(gt_tz) * 40
    #gt_tz = scipy.ndimage.median_filter(gt_tz, 3)
    lo_tz = scipy.ndimage.median_filter(lo_tz, 21) * 40
    hi_tz = scipy.ndimage.median_filter(hi_tz, 21) * 40

    l_gt = np.sqrt(gt_tz * gt_tz + gt_tx * gt_tx)
    l_lo = np.sqrt(lo_tz * lo_tz + lo_tx * lo_tx)
    l_hi = np.sqrt(hi_tz * hi_tz + hi_tx * hi_tx)

    x_axis = np.array(x_axis) / 40.0

    l_gt = scipy.ndimage.median_filter(l_gt, 51)
    for i in range(len(gt_tz)):
        if (l_lo[i] != 0):
            lo_tx[i] *= l_gt[i] / l_lo[i]
            lo_tz[i] *= l_gt[i] / l_lo[i]
        
        if (l_hi[i] != 0):
            hi_tx[i] *= l_gt[i] / l_hi[i]
            hi_tz[i] *= l_gt[i] / l_hi[i]

    # Compute errors
    ARPE = 0.0
    ARRE = 0.0
    AEE = 0.0
    for i in range(len(gt_tz)):
        AEE += math.hypot(gt_tz[i] - lo_tz[i], gt_tx[i] - lo_tx[i])
        ARRE += abs(gt_qy[i] - lo_qy[i])

        RPE_cos = (gt_tz[i] * lo_tz[i] + gt_tx[i] * lo_tx[i])
        if (l_gt[i] > 0):
            RPE_cos /= (l_gt[i] * l_gt[i])
        if (RPE_cos >  1.0): RPE_cos =  1.0
        if (RPE_cos < -1.0): RPE_cos = -1.0
        ARPE += math.acos(RPE_cos)

    print ("\n\nError scores:")
    print ("AEE:", AEE / len(gt_tz))
    print ("ARPE:", ARPE / len(gt_tz))
    print ("ARRE:", ARRE / len(gt_tz))
    print ("Length:", x_axis[-1], "sec.")

    # ==============

    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(X, Y, Z, linewidth=0) 
    #fig.tight_layout()

    #fig, axs = plt.subplots(6, 1)
    
    #plt.rcParams['axes.formatter.useoffset'] = False
    fig = plt.figure()
    gs = grd.GridSpec(6, 2, height_ratios=[1,1,1,1,1,1], width_ratios=[50,1], hspace=0.5, wspace=0.02)

    ax0 = plt.subplot(gs[0])
    im0 = ax0.imshow(Z_qy, origin='lower', interpolation='nearest', aspect='auto')
    ax0.set_title("Angular Speed")
    box = dict(facecolor='yellow', pad=5, alpha=0.2)
    ax0.set_ylabel('Class Id', bbox=box, labelpad=3)

    ax1 = plt.subplot(gs[1])
    cb1 = plt.colorbar(im0, cax = ax1)
    cb1.set_label('probability')

    ax2 = plt.subplot(gs[2])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    ax2.plot(x_axis, lo_qy, 'orange', lw=2.0, label='Prediction') 
    ax2.plot(x_axis, gt_qy, 'k', lw=0.5, label='Ground Truth')
    ax2.set_xlim(min(x_axis), max(x_axis))
    ax2.set_ylabel('[rad / sec.]', bbox=box, labelpad=3)
    #ax2.set_xlabel('time [sec.]', bbox=box)
    ax2.legend()


    ax4 = plt.subplot(gs[4])
    im4 = ax4.imshow(Z_tz, origin='lower', interpolation='nearest', aspect='auto')
    ax4.set_title("Linear Speed, Z axis")
    box = dict(facecolor='yellow', pad=5, alpha=0.2)
    ax4.set_ylabel('Class Id', bbox=box, labelpad=3)

    ax5 = plt.subplot(gs[5])
    cb5 = plt.colorbar(im4, cax = ax5)
    cb5.set_label('probability')

    ax6 = plt.subplot(gs[6])
    ax6.spines['right'].set_visible(False)
    ax6.spines['top'].set_visible(False)
    ax6.xaxis.set_ticks_position('bottom')
    ax6.yaxis.set_ticks_position('left')
    ax6.plot(x_axis, lo_tz, 'orange', lw=2.0, label='Prediction') 
    ax6.plot(x_axis, gt_tz, 'k', lw=0.5, label='Ground Truth')
    ax6.set_xlim(min(x_axis), max(x_axis))
    ax6.set_ylabel('[m / sec.]', bbox=box, labelpad=3)
    #ax6.set_xlabel('time [sec.]', bbox=box)
    ax6.legend()


    ax8 = plt.subplot(gs[8])
    im8 = ax8.imshow(Z_tx, origin='lower', interpolation='nearest', aspect='auto')
    ax8.set_title("Linear Speed, X axis")
    box = dict(facecolor='yellow', pad=5, alpha=0.2)
    ax8.set_ylabel('Class Id', bbox=box, labelpad=3)

    ax9 = plt.subplot(gs[9])
    cb9 = plt.colorbar(im8, cax = ax9)
    cb9.set_label('probability')

    ax10 = plt.subplot(gs[10])
    ax10.spines['right'].set_visible(False)
    ax10.spines['top'].set_visible(False)
    ax10.xaxis.set_ticks_position('bottom')
    ax10.yaxis.set_ticks_position('left')
    ax10.plot(x_axis, lo_tx, 'orange', lw=2.0, label='Prediction') 
    ax10.plot(x_axis, gt_tx, 'k', lw=0.5, label='Ground Truth')
    ax10.set_xlim(min(x_axis), max(x_axis))
    ax10.set_ylabel('[m / sec.]', bbox=box, labelpad=3)
    ax10.set_xlabel('time [sec.]', bbox=box)
    ax10.legend()

    fig.align_ylabels()

    
    #ax12 = plt.subplot(gs[12])
    #ax14 = plt.subplot(gs[14])

    #ax12.plot(x_axis, hash_y)
    #ax14.plot(x_axis, hash_x)
    #ax2.plot(x_axis, hi_qy)


    #im1 = axs[2].imshow(Z_tz, origin='lower')
    #axs[3].plot(x_axis, gt_tz)
    #axs[3].plot(x_axis, lo_tz) 
    #axs[3].plot(x_axis, hi_tz)

    #im2 = axs[4].imshow(Z_tx, origin='lower')
    #axs[5].plot(x_axis, gt_tx)
    #axs[5].plot(x_axis, lo_tx) 
    #axs[5].plot(x_axis, hi_tx)

    #axs[2].plot(x_axis, hash_x)
    #axs[3].plot(x_axis, hash_y)
    #axs[4].plot(x_axis, hash_z)
    #axs[5].plot(x_axis, hash_e)
    #axs[6].plot(x_axis, hash_p)
    #axs[7].plot(x_axis, hash_n)
    #axs[8].plot(x_axis, hash_g)
    #axs[5].plot(x_axis, hash_a)

    plt.show()
