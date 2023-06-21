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


def display(cloud):
    o3d.visualization.draw_geometries(cloud)


def downsampling(cloud, voxel_size=0.25, show=False):
    print("Downsampling")
    downpcd = cloud.voxel_down_sample(voxel_size=voxel_size)
    if show == True:
        display([downpcd])
    return downpcd


def remove_statistical_outlier(cloud, nb_neighbors=500, std_ratio=0.1, show=False):
    print("Statistical oulier removal")
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                               std_ratio=std_ratio)
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    if show == True:
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
        display([inlier_cloud, outlier_cloud])
    return (inlier_cloud, outlier_cloud)


def estimate_normals(cloud, radius=0.1, max_nn=30, show=False):
    print("Recompute the normal of the downsampled point cloud")
    cloud = cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    if show == True:
        display([cloud])
    return cloud


def PCDPointCloud(cloud, distance_threshold=0.05, ransac_n=3,
                  num_iterations=100, show=False):
    print("Start PCD point plane segmentation")
    plane_model, inliers = cloud.segment_plane(distance_threshold=distance_threshold,
                                               ransac_n=ransac_n,
                                               num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    inlier_cloud = cloud.select_by_index(inliers)
    outlier_cloud = cloud.select_by_index(inliers, invert=True)

    if show == True:
        #         inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud.paint_uniform_color([1.0, 0, 0])
        display([inlier_cloud, outlier_cloud])

    return (inlier_cloud, outlier_cloud)


def DBSCAN_clustering(cloud, eps=0.1, min_points=100, print_progress=False, show=False, pcd_ori=None):
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))

    max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

    if show == True:
        if pcd_ori != None:
            display([pcd_ori, cloud])
        else:
            display([cloud])
    return cloud, np.array(labels)


def DBSCAN_clustering_custom(cloud, eps=0.1, min_points=100, print_progress=False, show=False, pcd_ori=None):
    labels = np.array(
        cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))
    if len(labels) > 0:
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")

        if show == True:
            colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
            if pcd_ori != None:
                display([pcd_ori, cloud])
            else:
                display([cloud])
        return cloud, labels
    else:
        # print("point cloud has 0 clusters")
        return None, None


def select_by_np(cloud, labels, topcd=False, show=False):
    max_label = labels.max()
    _cloud = np.asarray(cloud.points)
    assert len(_cloud) == len(labels)

    display_temp = []
    for n in range(0, max_label):
        temp = []
        for _point, _label in zip(_cloud, labels):
            if _label == n:
                temp.append(_point)
        temp = np.asarray(temp)
        # print(n,"=" , len(temp))
        if topcd == True:
            pcd_temp = o3d.geometry.PointCloud()
            pcd_temp.points = o3d.utility.Vector3dVector(temp)
            display_temp.append(pcd_temp)
        else:
            display_temp.append(temp)
    if show == True:
        display(display_temp)
    return display_temp


def make_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)
    return None

def loadCloudFromTxt(inpath):
    '''
    Reload point clouds from txt files.
    :param inpath: The path of input data
    :return: o3d.geometrt.PointCloud object
    '''

    n = 0
    # temp =
    for fn in os.listdir(inpath):
        if n == 0:
            temp = np.loadtxt(os.path.join(inpath, fn))
        else:
            data = np.loadtxt(os.path.join(inpath, fn))
            temp = np.vstack((temp, data))
        n += 1

    print(f"Reloaded {n} clusters")
    pcd_temp = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(temp)

    return pcd_temp