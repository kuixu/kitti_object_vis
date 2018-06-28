# KITTI Object data transformation and visualization



## Dataset 

Download the data (calib, image\_2, label\_2, velodyne) from [Kitti Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and place it in your data folder at `kitti/object`


The folder structure is as following:
```
kitti
    object
        testing
        training
            calib
            image_2
            label_2
            velodyne
```

## Requrements

1. mayavi 
2. vtk

## Visualization

1. 3D boxes on LiDar point cloud in volumetric mode
2. 2D and 3D boxes on Camera image
3. 2D boxes on LiDar Birdview
4. LiDar data on Camera image


```python
python kitti_object.py /path/to/kitti/object
```

## Demo

#### 2D, 3D boxes LiDar data on Camera image
<img src="./rgb.png" alt="2D, 3D boxes LiDar data on Camera image" align="center" />
#### LiDar birdview and point cloud (3D) 
<img src="./lidar.png" alt="LiDar point cloud and birdview" align="center" />


## Acknowlegement

Code is mainly from [f-pointnet](https://github.com/charlesq34/frustum-pointnets) and [MV3D](https://github.com/bostondiditeam/MV3D)
