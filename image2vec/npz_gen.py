#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, signal, glob, math
import matplotlib.pyplot as plt
import pyquaternion as qt


def quaternion_to_euler(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))
    Z = math.atan2(t3, t4)

    return X, Y, Z 


def transform_pose(obj, cam):
    pos = obj[0] - cam[0]
    #inv_rot = cam[1].inverse
    inv_rot = cam[1]
    rotated_pos = inv_rot.rotate(pos)
    return [rotated_pos, obj[1] * cam[1].inverse]


def compute_vel(p1, p2):
    vel_t = (p2[0] - p1[0])
    vel_r = p2[1] * p1[1].inverse
    return [vel_t, vel_r]


def compute_vel_local(p1, p2):
    p2_ = transform_pose(p2, p1)
    p1_ = [np.array([0, 0, 0]), qt.Quaternion(1, 0, 0, 0)]
    return compute_vel(p1_, p2_)


def cam_poses_to_vels(cam_traj):
    ret = []
    last_pos = cam_traj[0]
    for i in range(len(cam_traj)):
        ret.append(compute_vel_local(last_pos, cam_traj[i]))
        last_pos = cam_traj[i]
    return ret


def undistort_img(img, K, D):
    Knew = K.copy()
    Knew[(0,1), (0,1)] = 0.87 * Knew[(0,1), (0,1)]
    img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    return img_undistorted


def read_calib(fname):
    K = np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])
    D = np.array([0.0, 0.0, 0.0, 0.0])

    lines = []
    with open(fname) as calib:
        lines = calib.readlines()

    # A single line: fx, fy, xc, cy, k1...k4
    if (len(lines) == 1):
        calib = lines[0].split(' ')
        K[0][0] = calib[0]
        K[1][1] = calib[1]
        K[0][2] = calib[2]
        K[1][2] = calib[3]
        D[0] = calib[4]        
        D[1] = calib[5]
        D[2] = calib[6]        
        D[3] = calib[7]
        return K, D

    K_txt = lines[0:3]
    D_txt = lines[4]
    
    for i, line in enumerate(K_txt):
        for j, num_txt in enumerate(line.split(' ')[0:3]):
            K[i][j] = float(num_txt)

    for j, num_txt in enumerate(D_txt.split(' ')[0:4]):
        D[j] = float(num_txt)

    return K, D


def trajectory_from_vel(Vx, Vy, Yaw):
    ret = []
    last_pos = [0, 0, 0]
    ret.append(last_pos)

    for i in range(len(Vx)):
        dx = Vx[i]
        dy = Vy[i]
        dyaw = -Yaw[i]

        yaw = last_pos[2] + dyaw
        px  = last_pos[0] + dx * math.cos(yaw) - dy * math.sin(yaw)
        py  = last_pos[1] + dx * math.sin(yaw) + dy * math.cos(yaw)

        last_pos = [px, py, yaw]
        ret.append(last_pos)

    return np.array(ret)


def get_index(cloud, index_w):
    idx = [0]
    if (cloud.shape[0] < 2):
        return np.array(idx, dtype=np.uint32)

    last_ts = cloud[0][0]
    for i, e in enumerate(cloud):
        while (e[0] - last_ts > index_w):
            idx.append(i)
            last_ts += index_w

    idx.append(cloud.shape[0] - 1)
    return np.array(idx, dtype=np.uint32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default='.',
                        required=False)
    parser.add_argument('--discretization',
                        type=float,
                        required=False,
                        default=0.01)

    args = parser.parse_args()

    print ("Opening", args.base_dir)    

    K, D = read_calib(args.base_dir + '/calib.txt')
    D /= 10.0

    print (K)
    print (D)

    print ("Poses...")
    timestamps = np.loadtxt(os.path.join(args.base_dir, 'depth_ts.txt'), usecols=1)
    poses = np.loadtxt(os.path.join(args.base_dir, 'poses.txt'))

    print (len(timestamps), "timestamps")
    print (len(poses), "poses")
    if (len(timestamps) != len(poses)):
        print ("Lengths do not match! Exiting...")
        sys.exit()

    cam_trajectory = []
    for i in range(poses.shape[0]):
        v = np.array([poses[i,9], poses[i,10], poses[i,11]])
        q = qt.Quaternion(matrix=np.array([[poses[i,0], poses[i,1], poses[i,2]],
                                           [poses[i,3], poses[i,4], poses[i,5]],
                                           [poses[i,6], poses[i,7], poses[i,8]]]))
        cam_trajectory.append([v, q])
    cam_vels = cam_poses_to_vels(cam_trajectory) # not scaled by ts

    gT_x = np.array([v[0][0] for v in cam_vels])
    gT_y = np.array([v[0][1] for v in cam_vels])
    gT_z = np.array([v[0][2] for v in cam_vels])
    gT_2 = np.array([v[0][0] * v[0][0] + v[0][1] * v[0][1] + v[0][2] * v[0][2] for v in cam_vels])
    gEuler = [quaternion_to_euler(v[1]) for v in cam_vels]
    gQ_x = np.array([q[0] for q in gEuler])
    gQ_y = np.array([q[1] for q in gEuler])
    gQ_z = np.array([q[2] for q in gEuler])

    gt_tj = trajectory_from_vel(gT_x, gT_z, gQ_y)
    plt.style.use('seaborn-whitegrid')
    plt.plot(-gt_tj[:,1], gt_tj[:,0], '-ok',
        markersize=1, linewidth=1,
        markerfacecolor='black',
        markeredgecolor='black',
        markeredgewidth=1
    )
    plt.savefig(os.path.join(args.base_dir, 'poses.svg'))

    print ("Reading the event file")
    cloud = np.loadtxt(args.base_dir + '/events.txt', dtype=np.float32)

    print ("Indexing")
    idx = get_index(cloud, args.discretization)

    print ("Saving...")
    np.savez_compressed(args.base_dir + "/recording.npz", events=cloud, index=idx, 
        discretization=args.discretization, K=K, D=D, poses=poses,
        Tx=gT_x, Ty=gT_y, Tz=gT_z, Qx=gQ_x, Qy=gQ_y, Qz=gQ_z,
        gt_ts=timestamps)

    print ("Done.")
