import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotly.express as px


import open3d as o3d

path = "/Users/karax/Desktop/1 Lab/2 Drone and parking/2Runs/pointcloud/tokoname_down/metashape_project/output/outputsecond"


def sortList(path):
    nlist = os.listdir(path)
    lenlist = list()

    for i in nlist:
        lenlist.append(int(len(np.loadtxt(os.path.join(path, i)))))

    nlist = np.array(nlist)
    lenlist = np.array(lenlist)

    df = pd.DataFrame({'file_name': nlist, 'cloud_len': lenlist})
    sorted_df = df.sort_values(by=['cloud_len'], ascending=False)
    return sorted_df


def visualization(fn):
    zdata = []
    xdata = []
    ydata = []
    data = np.loadtxt(fn)
    for l in data:
        x, y, z = l
        xdata.append(x)
        ydata.append(y)
        zdata.append(z)

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    zdata = np.array(zdata)
    print(f"File path: {fn} \n The number of points= {len(data)}")

    df = pd.DataFrame({'x_axis': xdata, 'y_axis': ydata, 'z_axis': zdata})

    ax = plt.axes(projection='3d')
    ax.scatter3D(xdata, ydata, zdata, c='b')
    ax.set_xlabel('X Axes')
    ax.set_ylabel('Y Axes')
    ax.set_zlabel('Z Axes')
    ax.set_title(fn.split("/")[-1:][0])
    plt.show()

    # fig = px.scatter_3d(df, x='x_axis', y='y_axis', z='z_axis')
    # fig.show()
    # points = np.array([xdata, ydata, zdata]).T
    # pcd_temp = o3d.geometry.PointCloud()
    # pcd_temp.points = o3d.utility.Vector3dVector(np.array([xdata, ydata, zdata]).T)

    #     # Point cloud normals calculation.
    # pcd_temp.estimate_normals()
    #         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # print(np.asarray(pcd_temp.normals))

    # Boinding box calculation.
    # aabb = pcd_temp.get_axis_aligned_bounding_box()
    # print(f"axis_aligned_bounding_box: {np.asarray(aabb)}")
    # aabb.color = [1, 0, 0]
    #
    # obb = pcd_temp.get_oriented_bounding_box()
    # print(f"oriented_bounding_box: {np.asarray(obb)}")
    # obb.color = [0, 0, 1]

    # Visualization
    # o3d.visualization.draw_geometries([pcd_temp
    #                                       , aabb, obb
    #                                    ])


# Launch
sorted_df = sortList(path)
print(sorted_df)
nfiles = len(sorted_df)
print(f"The number of files {nfiles}")
for line in range(nfiles):
    indexnb = line
    fn = sorted_df['file_name'].iloc[indexnb]

    file_path = os.path.join(path, fn)
    visualization(file_path)