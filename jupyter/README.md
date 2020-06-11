# Install remotely on a Ubuntu 16.04 server and visulize by Jupyter Notebook
## Install mayavi
- remote servers usually lack GUI supports, so first some libraries and backends should be installed
```
(base)$ sudo apt-get update && sudo apt-get install libgl1-mesa-glx xvfb
```
- install like locally, add [xvfbwrapper](https://github.com/enthought/mayavi/issues/477#issuecomment-477653210)
```
(base)$ conda create -n kitti_vis python=3.7
(base)$ conda activate kitti_vis
(kitti_vis)$ pip install opencv-python pillow scipy xvfbwrapper
(kitti_vis)$ conda install mayavi -c conda-forge
```
- install jupyter notebook
```
(kitti_vis)$ pip install notebook
```
- install ipywidgets and ipyevents, enable them as notebook extensions
```
(kitti_vis)$ pip install ipywidgets ipyevents
(kitti_vis)$ jupyter nbextension enable --py --sys-prefix widgetsnbextension
(kitti_vis)$ jupyter nbextension enable --py --sys-prefix ipyevents
```
- install and enable mayavi extension
```
(kitti_vis)$ jupyter nbextension install --py --sys-prefix mayavi
(kitti_vis)$ jupyter nbextension enable --py --sys-prefix mayavi
```
- to test on Jupyter notebook, first set [ETS_TOOLKIT](https://github.com/enthought/mayavi/issues/439#issuecomment-251703994)
```
(kitti_vis)$ export ETS_TOOLKIT='null'
```
- then you can test mayavi intallation by [this notebook](test_mayavi.ipynb), which should show an interactive 3D curve