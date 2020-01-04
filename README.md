# Sparse Data Fusion for Scene Segmentation
## Overview
This repository contains Jupyter notebooks and Python files for sparse data fusion for semantic segmentation of lidar point clouds fused with co-registered RGB imagery.  This work was implemented using PyTorch and [PointNet++](https://github.com/charlesq34/pointnet2), a framework that extends the capabilities of the original PointNet DCNN framework through the use of multi-scale feature analysis.  The dataset we chose to analyze for this project was Audi's [A2D2](https://www.audi-electronics-venture.de/aev/web/en/driving-dataset/dataset.html), which contains over 40000 scenes taken from an autonomous vehicle.  Each scene contains 3D xyz point cloud data (lidar), as well as 2D RGB data, semantically-segmented 2D ground truth data, and a mapping between 3D point space and 2D RGB space.

My final report for this project can be found under [report.pdf](https://github.com/rmsander/sparse-fusion-scene-seg/blob/master/report.pdf).

 ## Environment Installation
 An Anaconda environment was used for this project.  To set up the conda environment, you can do so with the following bash command:
 
 `conda env create -f environment.yml`
 
 ## Port Forwarding for Remote Jupyter Notebooks
 To use Jupyter notebooks on a remote host (e.g. AWS EC2), see the bash script `AWS_ssh.sh` and replace the `ami_key.pem` and `remote_user@remote_host` placeholders with your relevant remote host key (if applicable) and remote machine, respectively.  This script forwards remote Jupyter notebooks from the 9999 port (default) to the 8888 port (in case 9999 is being used for a local Jupyter notebook).  
 
 If you prefer, you can run the `AWS_ssh.sh` bash script to ssh into your remote host and set the remote port for Jupyter notebooks to 8888.  Once you have ssh'ed into your remote machine, you can type the following command (note: you must have Jupyter notebook already installed) to start a Jupyter notebook that can be accessed from your local browser:
 
 `jupyter notebook --no-browser`
 
 After typing this, you should see a URL for your Jupyter notebook in the prompt.  Copy and paste this URL into your local browser, and replace the port number `9999` with `8888`.  This should enable you to access your remote Jupyter notebook session from your local browser.
 
 ## Usage 
 The Jupyter notebooks and Python files in this repository can be used to pre-process this dataset, create different datasets, and train and test neural network frameworks for determining the capabilities of PointNet++ in the task of using xyz-RGB data for semantic segmentation.  
 
### Jupyter Notebooks for Remote Development
The Jupyter notebooks may be especially helpful for remote development.  They primarily contain code for pre-processing the dataset, creating variants of the dataset, and creating Python objects for PyTorch Datasets and DataLoaders.  These notebooks were written for use for the A2D2 dataset, but the methods and code blocks contained within these notebooks may be extendable to different datasets/different applications of the A2D2 dataset as well.  The main Jupyter notebooks for this project, for pre-processing data and creating useful datasets, are:

1. `audi_dataset_pre_processing.ipynb`
2. `setup_training.ipynb`

### Python Files for Screen Sessions
Some of the Jupyter notebooks used in this repository were also converted into Python files:

1. `audi_dataset_pre_processing.py`
2. `create_partitioned_dataset.py`
3. `neural_net_testing.py`


This proved useful for running code for extended durations, in which hyper-parameters had to be changed.

### Training, Testing, and Model Code
Code for training, testing, and the PointNet++ model can be found in the `PointNet2/` directory in this repository.  Most of the files in this framework are from [Charles Qi's PointNet++ PyTorch implementaton](https://github.com/charlesq34/pointnet2), with the following file additions:

1. `pad_collate_fn.py` (Contains code for my "Resample Via Padding" custom collation function for batched training.)
2. `train_semseg-focal_loss.py` (Training code with focal loss.)
3. `train_semseg-transfer_learning.py` (Training code with self-incremental transfer learning.)
4. `data_utils/A2D2DataLoader.py` (PyTorch DataLoader class for interfacing with A2D2 dataset.)
 
## Acknowledgements
Thank you to the 6.869 team at MIT for providing me with crucial guidance and AWS resources for this project, the Audi Electronics Venture Group for open access to the A2D2 dataset, and to Charles Qi for providing open access to the PointNet++ architecture.
 

