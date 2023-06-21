#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:54:50 2022

@author: Jiaming Liu
"""

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import shutil
import time
from utils import *
from myparas import getparas

if __name__ == "__main__":

    # Path
    path = getparas.mypath()
    os.chdir(path)
    fn = os.listdir()
    # Make a path for save
    make_path('outputfirst')

    # # Parameters
    hpparas = getparas.myparas()
    Downsampling_voxel_size = hpparas['Downsampling_voxel_size']

    # Key parameters
    Plane_seg_threshold = hpparas['Plane_seg_threshold']
    Refine_size = hpparas['Refine_size']
    Refine_times = hpparas['Refine_times']
    clustering_eps = hpparas['clustering_eps']
    Ransac_n = hpparas['Ransac_n']
    OriShow = hpparas['OriShow']


    # Define workflow
    def workflow(pcd):
        print(f"File= {line}, Refine_times= 0")
        pcd = downsampling(pcd, voxel_size=Downsampling_voxel_size, show=False)
        pcd = remove_statistical_outlier(pcd, nb_neighbors=50, std_ratio=2,
                                         show=False)[0]
        if len(np.array(pcd.points)) >= Ransac_n:
            pcd = PCDPointCloud(pcd, distance_threshold=Plane_seg_threshold,
                                ransac_n=Ransac_n,
                                num_iterations=10000, show=False)[1]
            pcd, labels = DBSCAN_clustering_custom(pcd, eps=clustering_eps,
                                                   min_points=3,
                                                   print_progress=False,
                                                   show=False, pcd_ori=None)
            clouds_ = select_by_np(pcd, labels, topcd=False, show=False)
        else:
            clouds_ = [np.array(0)]

        return clouds_  # list(nparray)


    def refine_process(clouds, refine_size=2000, show=False):
        temp = []
        # n=0
        for _cloud in clouds:
            len_c = len(_cloud)
            if len_c > refine_size:
                pcd_temp = o3d.geometry.PointCloud()
                pcd_temp.points = o3d.utility.Vector3dVector(_cloud)
                _pcd = PCDPointCloud(pcd_temp,
                                     distance_threshold=Plane_seg_threshold,
                                     ransac_n=Ransac_n,
                                     num_iterations=1000, show=False)[1]
                if len(np.asarray(_pcd.points)) > 0:
                    _pcd, labels = DBSCAN_clustering_custom(_pcd,
                                                            eps=clustering_eps,
                                                            min_points=3,
                                                            print_progress=False,
                                                            show=False,
                                                            pcd_ori=None)
                    if _pcd != None:
                        _clouds = select_by_np(_pcd, labels, topcd=False, show=False)
                        for _cloud in _clouds:
                            temp.append(_cloud)
            else:
                # print(len_c, "not refine")
                temp.append(_cloud)

            # n+=1
            # print(n)

        if show == True:
            display(temp)
        return temp


    # Launch
    pcdL = list()
    nb = 0

    print("Start")
    for line in fn:
        if os.path.splitext(line)[1] == '.pcd':
            # Load original file
            print(f"Load original file= {line}")
            if OriShow:
                pcd_ori = o3d.io.read_point_cloud(line)
                pcd_ori = downsampling(pcd_ori, voxel_size=Downsampling_voxel_size,
                                       show=False)
                pcdL.append(pcd_ori)

            # Load work file
            pcd = o3d.io.read_point_cloud(line)
            if len(np.asarray(pcd.points)) > 0:
                # main part
                clouds = workflow(pcd)
                # refine part
                if Refine_times == -1:
                    i = 0
                    while np.any(np.array([len(x) for x in clouds]) >= Refine_size):
                        print(f"File= {line}, Refine_times= {i + 1}")
                        clouds = refine_process(clouds, refine_size=Refine_size)
                        i += 1
                        time.sleep(1)

                elif Refine_times >= 0:
                    for i in range(Refine_times):
                        print(f"File= {line}, Refine_times= {i + 1}")
                        clouds = refine_process(clouds, refine_size=Refine_size)
                        time.sleep(1)
                else:
                    print("With out refine, please set it >= -1")

                print(f"{i + 1} times plane segmentation")

                # Save files
                pcd_temp_total = []
                for _cloud in clouds:
                    # save
                    np.savetxt(f"outputfirst/distress {nb}.txt",
                               np.asarray(_cloud)
                               # ,delimiter=","
                               )
                    pcd_temp = o3d.geometry.PointCloud()
                    pcd_temp.points = o3d.utility.Vector3dVector(_cloud)
                    pcdL.append(pcd_ori)
                    pcd_temp_total.append(pcd_temp)
                    nb += 1

    print(f"total {nb} distress")
    # display(pcd_temp_total)
