# 仿真数据　to DB
使用一组仿真场景来生成相机与观测；
1. 调用vio-simulation中生成数据的方法，生成：all_points.txt; camera_pose.txt;以及每个图像的观测点信息：keyframe/all_points_n.txt
2. 调用database.py，生成db文件。

# 在2D上画出三维点与相机
1. 运行visualize.py