import os
import sys
import collections
import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from read_model import ReadModel


_if_show_3d = True
_if_cam_pose = False

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
    ax.set_aspect('equal')
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

def get_residuals(cameras, images, points3D):
    print "optimization residuals: "
    read_model = ReadModel()
    point = points3D[1]
    point2d_id = point.point2D_idxs[0]
    image_id = point.image_ids[0]
    image = images[image_id]
    pixel = image.xys[point2d_id]

    p_G = point.xyz
    R = read_model.qvec2rotmat(image.qvec)
    if _if_cam_pose:
        # global to local:  p_G = R*p_L + T;
        p_G_t = (p_G - image.tvec)
        p_L = np.dot(R.transpose(), np.reshape(p_G_t,[3,1]))
    else:
        #
        p_L = np.dot(p_G, R) + image.tvec
    # local to pixel;
    f = cameras[1].params[0]
    cx = cameras[1].params[1]
    cy = cameras[1].params[2]
    x_pi = f * p_L[0]/p_L[2] + cx
    y_pi = f * p_L[1]/p_L[2] + cy
    x_resi = x_pi - pixel[0]
    y_resi = y_pi - pixel[1]


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

    #visulize(cameras, images, points3D)
    get_residuals(cameras, images, points3D)


if __name__ == "__main__":
    main()
