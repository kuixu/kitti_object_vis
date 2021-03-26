# KITTI Object data transformation and visualization



## Dataset

Download the data (calib, image\_2, label\_2, velodyne) from [Kitti Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and place it in your data folder at `kitti/object`


The folder structure is as following:
```
kitti
    object
        testing
            calib
               000000.txt
            image_2
               000000.png
            label_2
               000000.txt
            velodyne
               000000.bin
            pred
               000000.txt
        training
            calib
               000000.txt
            image_2
               000000.png
            label_2
               000000.txt
            velodyne
               000000.bin
            pred
               000000.txt
```

## Install locally on a Ubuntu 16.04 PC with GUI
- start from a new conda enviornment:
```
(base)$ conda create -n kitti_vis python=3.7 # vtk does not support python 3.8
(base)$ conda activate kitti_vis
```
- opencv, pillow, scipy, matplotlib
```
(kitti_vis)$ pip install opencv-python pillow scipy matplotlib
```
- install mayavi from conda-forge, this installs vtk and pyqt5 automatically
```
(kitti_vis)$ conda install mayavi -c conda-forge
```
- test installation
```
(kitti_vis)$ python kitti_object.py --show_lidar_with_depth --img_fov --const_box --vis
```

**Note: the above installation has been tested not work on MacOS.**

## Install remotely
Please refer to the [jupyter](jupyter/) folder for installing on a remote server and visulizing in Jupyter Notebook.

## Visualization

1. 3D boxes on LiDar point cloud in volumetric mode
2. 2D and 3D boxes on Camera image
3. 2D boxes on LiDar Birdview
4. LiDar data on Camera image


```shell
$ python kitti_object.py --help
usage: kitti_object.py [-h] [-d N] [-i N] [-p] [-s] [-l N] [-e N] [-r N]
                       [--gen_depth] [--vis] [--depth] [--img_fov]
                       [--const_box] [--save_depth] [--pc_label]
                       [--show_lidar_on_image] [--show_lidar_with_depth]
                       [--show_image_with_boxes]
                       [--show_lidar_topview_with_boxes]

KIITI Object Visualization

optional arguments:
  -h, --help            show this help message and exit
  -d N, --dir N         input (default: data/object)
  -i N, --ind N         input (default: data/object)
  -p, --pred            show predict results
  -s, --stat            stat the w/h/l of point cloud in gt bbox
  -l N, --lidar N       velodyne dir (default: velodyne)
  -e N, --depthdir N    depth dir (default: depth)
  -r N, --preddir N     predicted boxes (default: pred)
  --gen_depth           generate depth
  --vis                 show images
  --depth               load depth
  --img_fov             front view mapping
  --const_box           constraint box
  --save_depth          save depth into file
  --pc_label            5-verctor lidar, pc with label
  --show_lidar_on_image
                        project lidar on image
  --show_lidar_with_depth
                        --show_lidar, depth is supported
  --show_image_with_boxes
                        show lidar
  --show_lidar_topview_with_boxes
                        show lidar topview
  --split               use training split or testing split (default: training)

```

```shell
$ python kitti_object.py
```
Specific your own folder,
```shell
$ python kitti_object.py -d /path/to/kitti/object
```

Show LiDAR only
```
$ python kitti_object.py --show_lidar_with_depth --img_fov --const_box --vis
```

Show LiDAR and image
```
$ python kitti_object.py --show_lidar_with_depth --img_fov --const_box --vis --show_image_with_boxes
```

Show LiDAR and image with specific index
```
$ python kitti_object.py --show_lidar_with_depth --img_fov --const_box --vis --show_image_with_boxes --ind 1 
```

Show LiDAR with `modified LiDAR file` with an additional point cloud label/marker as the 5th dimention(5 vector: x, y, z, intensity, pc_label). (This option is for very specific case. If you don't have this type of data, don't use this option).
```
$ python kitti_object.py --show_lidar_with_depth --img_fov --const_box --vis --pc_label
```

## Demo

#### 2D, 3D boxes and LiDar data on Camera image
<img src="./imgs/rgb.png" alt="2D, 3D boxes LiDar data on Camera image" align="center" />
<img src="./imgs/lidar-label.png" alt="boxes with class label" align="center" />
Credit: @yuanzhenxun

#### LiDar birdview and point cloud (3D)
<img src="./imgs/lidar.png" alt="LiDar point cloud and birdview" align="center" />

## Show Predicted Results

Firstly, map KITTI official formated results into data directory
```
./map_pred.sh /path/to/results
```

```python
python kitti_object.py -p --vis
```
<img src="./imgs/pred.png" alt="Show Predicted Results" align="center" />


## Acknowlegement

Code is mainly from [f-pointnet](https://github.com/charlesq34/frustum-pointnets) and [MV3D](https://github.com/bostondiditeam/MV3D)
