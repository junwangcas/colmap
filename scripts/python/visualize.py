import os
import sys
import collections
import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from read_model import ReadModel


_if_show_3d = True

def visulize(cameras, images, points3D):
    print "visulize in 2D: "
    # points
    size_points = len(points3D)
    points_xyz = np.zeros([size_points, 3])
    for i in range(1, (len(points3D)+1)):
        point3d = points3D[i]
        points_xyz[i-1, :] = point3d.xyz
    # cameras;
    size_images = len(images)
    images_xyz = np.zeros([size_images, 3])
    for i in range(1, len(images) + 1):
        image = images[i]
        images_xyz[i-1, :] = image.tvec
    # draw
    fig, ax = plt.subplots()
    ax.scatter(points_xyz[:, 0], points_xyz[:, 2], label='points')
    ax.legend()
    plt.scatter(images_xyz[:, 0], images_xyz[:, 2], color = 'r', label='images')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.title('points in 2D')


    if _if_show_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(images_xyz[:,0],images_xyz[:,1],images_xyz[:,2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title('points in 3D')
    plt.show()

def main():
    path = ''
    ext = ''
    if len(sys.argv) != 3:
        path = '/media/nvidia/TWOTB/work_code/colmap/cmake-build-debug/src/exe/model_save'
        ext = '.txt'
    else:
        path = sys.argv[1]
        ext = sys.argv[2]

    read_model = ReadModel()
    cameras, images, points3D = read_model.read_model(path, ext)

    print("num_cameras:", len(cameras))
    print("num_images:", len(images))
    print("num_points3D:", len(points3D))

    visulize(cameras, images, points3D)


if __name__ == "__main__":
    main()
