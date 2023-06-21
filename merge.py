#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:54:50 2022

@author: Jiaming Liu
"""

from utils import *
from myparas import getparas

if __name__ == "__main__":

    # Make path
    path = getparas.mypath()
    inpath = os.path.join(path, "outputfirst")
    outpath = os.path.join(path, "outputsecond")
    os.chdir(path)
    make_path(outpath)

    # Load parameters
    Downsampling_voxel_size = getparas.myparas()['Downsampling_voxel_size']

    # Merge process
    pcd_temp = loadCloudFromTxt(inpath)
    pcd_temp, _ = remove_statistical_outlier(pcd_temp, nb_neighbors=250, std_ratio=0.1,
                                             show=False)
    pcd_temp, labels = DBSCAN_clustering(pcd_temp, eps=0.1, min_points=10, show=False)
    clouds = select_by_np(pcd_temp, labels, topcd=False, show=False)

    # Display
    displayList = []
    for line in os.listdir(path):
        if os.path.splitext(line)[1] == '.pcd':
            original = o3d.io.read_point_cloud(os.path.join(path, line))
            original = original.voxel_down_sample(voxel_size=Downsampling_voxel_size)
            displayList.append(original)
    displayList.append(pcd_temp)
    display(displayList)

    # Save merged clouds
    nb = 0
    for _cloud in clouds:
        np.savetxt(f"outputsecond/distress {nb}.txt", np.asarray(_cloud))
        nb += 1
    print(f"After have {nb} distress")