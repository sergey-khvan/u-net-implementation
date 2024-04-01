# u-net implementation
![Static Badge](https://img.shields.io/badge/Docker-gray?style=flat-square&logo=docker&logoColor=white)
![Static Badge](https://img.shields.io/badge/python-gray?style=flat-square&logo=python&logoColor=white)
![Static Badge](https://img.shields.io/badge/PyTorch-gray?style=flat-square&logo=pytorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/OpenCV-gray?style=flat-square&logo=opencv&logoColor=white)
![Static Badge](https://img.shields.io/badge/albumentations-gray?style=flat-square&logo=A&logoColor=white)

## Information

This project is a precise implementation of the u-net architecture from the [original paper](https://arxiv.org/abs/1505.04597v1).
Therefore, the majority of the hyperparameter choices were based on the choices reported in the paper.

## Setup

### 1. Clone this repository 
```
git clone https://github.com/sergey-khvan/u-net-implementation 
```
### 2. Set up a few directories

To do it, open the cloned directory and run the following in the terminal.
```
cd train/data
mkdir test train
cd test
mkdir images masks
cd ../train
mkdir images masks
cd ../..
```
### 3. Set up your data
If you want to use the dataset from the original paper, simply run the __unpack_tiffs.py.__

You can do it directly or by running : 
```
python unpack_tiff.py
```

You will also need to create two directories for saving the model checkpoints and end weights.
```
mkdir weights checkpoints
```

### 4. Build a Docker container
From the directory with Dockerfile (initial dir) 
```
docker build -t unet-image .
```

### 5. Run your container
Run in detached mode.
```
docker run -d \
  --name "unet-container" \
  unet-image \
  tail -f /dev/null
```
### 6. Interact with the container
```
docker exec -it unet-container /bin/bash
```
## Model

<p align="center">
  <img style="border-radius: 1%" src="assets/u-net-architecture.png" width="500" alt="accessibility text">
</p>

## Training

* __1000 epochs__ (~~original: 10 hours on Nvidia Titan GPU 6Gb~~)
* Loss function: Cross Entropy Loss
* Optimizer: Stochastic Gradient Descent
* Momentum: 0.99
* Batch size: 1

## Results

### Benchmarks from the original paper

|               | Warping Error | Rand Error | Pixel Error |
| :---          |    :----:     |      :---: |---:         |
| Implementation| 0.000833      | 0.12075    | 0.075       |
| Paper         | 0.000353      | 0.0382     |  0.0611     |

### Loss vs epoch graph
<p align="center">
  <img style="border-radius: 1%" src="assets/Losses.png" width="500" alt="accessibility text">
</p>

Few runs on [Weights and Biases](https://wandb.ai/sergey-khvan/u-net?nw=nwusersergeykhvan)
