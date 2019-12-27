# -*- coding: utf-8 -*-
# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

# This script is based on an original implementation by True Price.


import sys
import sqlite3
import numpy as np
import os
import argparse

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H)))

class simulation_data_reading():
    dir_prefix = "/media/nvidia/TWOTB/vins/vio_data_simulation/bin/"
    file_all_points = dir_prefix + "all_points.txt"
    file_camera_pose = dir_prefix + "cam_pose.txt"
    dir_keyframe_obs_prefix = dir_prefix + "keyframe/" + "all_points_"
    #file_db = "./data/database.db"
    file_db = "/media/nvidia/TWOTB/work_code/colmap/cmake-build-debug/src/exe/database.db"
    qw_index = 1
    size_cam = 0
    _all_points = []
    _cam_poses = []
    _all_observations = []
    ### camera setings;
    _image_height = 480
    _image_width = 640
    if_reduce_cam_size = False
    _size_cam_control = 25

    def run_read_data(self):
        self.readpoints()
        self.read_cam_poses()
        self.read_observation_one()

    def readpoints(self):
        with open(self.file_all_points, "r") as file_handle:
            lines = file_handle.readlines()
            for line in lines:
                line_split = line.split()
                line_float = map(float, line_split)
                point = [line_float[0], line_float[1], line_float[2]]
                self._all_points.append(point)
            print("read %d points: " % len(self._all_points))

    def read_observation_one(self):
        count_cam = 0
        while count_cam < self.size_cam:
            file_name = self.dir_keyframe_obs_prefix + str(count_cam) + ".txt"
            with open(file_name, "r") as file_handle:
                observation = []
                lines = file_handle.readlines()
                line_count = 0
                for line in lines:
                    line_split = line.split()
                    line_float = map(float, line_split)
                    point_obs = [line_float[4], line_float[5]]
                    observation.append(point_obs)
                    point = self._all_points[line_count]
                    assert line_float[0] == point[0]
                    assert line_float[1] == point[1]
                    assert line_float[2] == point[2]
                    line_count += 1
                self._all_observations.append(observation)
            count_cam += 1
        print("read %d observations" % len(self._all_observations))

    def read_cam_poses(self):
        with open(self.file_camera_pose, "r") as file_handle:
            lines = file_handle.readlines()
            for line in lines:
                line_split = line.split()
                line_float = map(float, line_split)
                pose = [line_float[self.qw_index], line_float[self.qw_index + 1], line_float[self.qw_index + 2],
                        line_float[self.qw_index + 3], line_float[self.qw_index + 4], line_float[self.qw_index + 5],
                        line_float[self.qw_index + 6]]
                self._cam_poses.append(pose)
            self.size_cam = len(self._cam_poses)
            print("read %d poses: " % len(self._cam_poses))
            if (self.if_reduce_cam_size):
                self.size_cam = self._size_cam_control

    def write_to_db(self):
        if os.path.exists(self.file_db):
            print("db exist and will remove it")
            os.remove(self.file_db)
        # Open the database.
        # /// 首先要连接olmap的数据库
        db = COLMAPDatabase.connect(self.file_db)

        # For convenience, try creating all the tables upfront.

        db.create_tables()

        # Create dummy cameras.
        # ///@todo model 是指的什么，params指的什么？
        # 构造两组数据，指定 宽 高 参数
        model1, width1, height1, params1 = \
            0, self._image_width, self._image_height, np.array((self._image_width, self._image_height, 384.))

        # 将相机模型首先增加到数据库中。
        camera_id1 = db.add_camera(model1, width1, height1, params1)

        # 利用相机模型生成若干的图像。
        image_ids = []
        count = 0
        while count < self.size_cam:
            image_name = "image%d.png" % count
            image_id = db.add_image(image_name, camera_id1)
            image_ids.append(image_id)
            count += 1
        # Create dummy keypoints.
        #
        # Note that COLMAP supports:
        #      - 2D keypoints: (x, y)
        #      - 4D keypoints: (x, y, theta, scale)
        #      - 6D affine keypoints: (x, y, a_11, a_12, a_21, a_22)
        # 关键点增加到图像上去；
        count = 0
        while count < self.size_cam:
            image_id = image_ids[count]
            keypoint = self._all_observations[count]
            keypoint = np.asarray(keypoint)
            db.add_keypoints(image_id, keypoint)
            count += 1

        matches_cam_keypt = []
        # 增加关键点之间的匹配关系；
        for i in range(self.size_cam - 1):
            for j in range(i + 1, self.size_cam):
                size_obs = len(self._all_observations[i])
                matches = np.arange(size_obs*2).reshape([size_obs, 2])
                increament_num = np.arange(size_obs)
                matches[:,0] = increament_num
                matches[:,1] = increament_num
                image_id1 = image_ids[i]
                image_id2 = image_ids[j]
                db.add_matches(image_id1, image_id2, matches)
                db.add_two_view_geometry(image_id1, image_id2, matches)
                matches_cam_keypt.append(matches)

        # Commit the data to the file.
        # 将他们之间的关系写入；
        db.commit()

        # Read and check cameras.

        rows = db.execute("SELECT * FROM cameras")

        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == camera_id1
        assert model == model1 and width == width1 and height == height1
        assert np.allclose(params, params1)

        # Read and check keypoints.

        keypoints = dict(
            (image_id, blob_to_array(data, np.float32, (-1, 2)))
            for image_id, data in db.execute(
                "SELECT image_id, data FROM keypoints"))

        assert np.allclose(keypoints[image_ids[0]], self._all_observations[0])

        # Read and check matches.

        pair_ids = [image_ids_to_pair_id(*pair) for pair in
                    ((image_ids[0], image_ids[1]),
                     (image_ids[0], image_ids[2]))]

        matches = dict(
            (pair_id_to_image_ids(pair_id),
             blob_to_array(data, np.uint32, (-1, 2)))
            for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
        )

        assert np.all(matches[(image_ids[0], image_ids[1])] == matches_cam_keypt[0])
        # Clean up.
        db.close()


def example_usage():
    ### /// 样例使用，数据库的地址
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    if os.path.exists(args.database_path):
        print("ERROR: database path already exists -- will not modify it.")
        return

    # Open the database.
    #/// 首先要连接olmap的数据库
    db = COLMAPDatabase.connect(args.database_path)

    # For convenience, try creating all the tables upfront.

    db.create_tables()

    # Create dummy cameras.
    #///@todo model 是指的什么，params指的什么？
    #构造两组数据，指定 宽 高 参数
    model1, width1, height1, params1 = \
        0, 1024, 768, np.array((1024., 512., 384.))
    model2, width2, height2, params2 = \
        2, 1024, 768, np.array((1024., 512., 384., 0.1))

    # 将相机模型首先增加到数据库中。
    camera_id1 = db.add_camera(model1, width1, height1, params1)
    camera_id2 = db.add_camera(model2, width2, height2, params2)

    # Create dummy images.
    # 利用相机模型生成若干的图像。
    image_id1 = db.add_image("image1.png", camera_id1)
    image_id2 = db.add_image("image2.png", camera_id1)
    image_id3 = db.add_image("image3.png", camera_id2)
    image_id4 = db.add_image("image4.png", camera_id2)

    # Create dummy keypoints.
    #
    # Note that COLMAP supports:
    #      - 2D keypoints: (x, y)
    #      - 4D keypoints: (x, y, theta, scale)
    #      - 6D affine keypoints: (x, y, a_11, a_12, a_21, a_22)
    # 生成若干个关键点；
    num_keypoints = 1000
    keypoints1 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints2 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints3 = np.random.rand(num_keypoints, 2) * (width2, height2)
    keypoints4 = np.random.rand(num_keypoints, 2) * (width2, height2)
    # 将这些关键点增加到图像上去；
    db.add_keypoints(image_id1, keypoints1)
    db.add_keypoints(image_id2, keypoints2)
    db.add_keypoints(image_id3, keypoints3)
    db.add_keypoints(image_id4, keypoints4)

    # Create dummy matches.
    # 增加关键点之间的匹配关系；
    M = 50
    matches12 = np.random.randint(num_keypoints, size=(M, 2))
    matches23 = np.random.randint(num_keypoints, size=(M, 2))
    matches34 = np.random.randint(num_keypoints, size=(M, 2))

    db.add_matches(image_id1, image_id2, matches12)
    db.add_matches(image_id2, image_id3, matches23)
    db.add_matches(image_id3, image_id4, matches34)
    db.add_two_view_geometry(image_id1, image_id2, matches12)
    db.add_two_view_geometry(image_id2, image_id3, matches23)
    db.add_two_view_geometry(image_id3, image_id4, matches34)




    # Commit the data to the file.
    # 将他们之间的关系写入；
    db.commit()

    # Read and check cameras.

    rows = db.execute("SELECT * FROM cameras")

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id1
    assert model == model1 and width == width1 and height == height1
    assert np.allclose(params, params1)

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id2
    assert model == model2 and width == width2 and height == height2
    assert np.allclose(params, params2)

    # Read and check keypoints.

    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints"))

    assert np.allclose(keypoints[image_id1], keypoints1)
    assert np.allclose(keypoints[image_id2], keypoints2)
    assert np.allclose(keypoints[image_id3], keypoints3)
    assert np.allclose(keypoints[image_id4], keypoints4)

    # Read and check matches.

    pair_ids = [image_ids_to_pair_id(*pair) for pair in
                ((image_id1, image_id2),
                 (image_id2, image_id3),
                 (image_id3, image_id4))]

    matches = dict(
        (pair_id_to_image_ids(pair_id),
         blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )

    assert np.all(matches[(image_id1, image_id2)] == matches12)
    assert np.all(matches[(image_id2, image_id3)] == matches23)
    assert np.all(matches[(image_id3, image_id4)] == matches34)

    # Clean up.

    db.close()
    # 这里是将刚才生成的数据删除。
    # if os.path.exists(args.database_path):
    #     os.remove(args.database_path)

def simulation_to_db():
    sim_reading = simulation_data_reading()
    sim_reading.run_read_data()
    sim_reading.write_to_db()

if __name__ == "__main__":
    if_use_simulation = True
    if if_use_simulation:
        # 读取simulation的结果，将其写到
        simulation_to_db()
    else:
        # 这个示例为官方自带的示例；
        example_usage()
