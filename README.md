#  Breathing Life into Faces: Speech-driven 3D Facial Animation with Natural Head Pose and Detailed Shape
This repository contains a pytorch implementation of "Breathing Life into Faces: Speech-driven 3D Facial Animation with Natural Head Pose and Detailed Shape"


> <a href='https://arxiv.org/abs/2310.20240'><img src='https://img.shields.io/badge/arXiv-2301.02379-red'></a> <a href='https://weizhaomolecules.github.io/VividTalker/'><img src='https://img.shields.io/badge/Project-Video-Green'></a>

![](src/data/teaser.gif)

This codebase provides:
- train code
- test code
- dataset
- checkpoints

The main sections are:
- Overview
- Instalation
- Download Data and Models
- Training from Scratch
- Testing with Pretrained Models
- visualization code for rendering

# Overview:
We provide models and code to train and test our listener motion models.

See below for sections:
- **Installation**: environment setup and installation for visualization
- **Download data and models**: download annotations and pre-trained models
- **Training from scratch**: scripts to get the training pipeline running from scratch
- **Testing with pretrianed models**: scripts to test pretrained models and save output motion parameters

## Installation:
Tested with cuda/9.0, cudnn/v7.0-cuda.9.0, and python 3.6.11

```
git clone git@github.com:evonneng/learning2listen.git

cd Vividtalker/src/
conda create -n vividtalker python=3.6
conda activate venv_l2l
pip install -r requirements.txt

export L2L_PATH=`pwd`
```

IMPORTANT: After installing torch, please make sure to modify the `site-packages/torch/nn/modules/conv.py` file by commenting out the `self.padding_mode != 'zeros'` line to allow for replicated padding for ConvTranspose1d as shown [here](https://github.com/NVIDIA/tacotron2/issues/182). 



# Training from Scratch:
Training a model from scratch follows a 2-step process.

1. Training VQ-VAEs for mouth movment and head pose:
```
cd src/vqgan/
sh train_pose.sh
sh train_exp.sh
```

2. Training decoders for mouth movement and head pose synthesis 
```
cd src
sh train_decoder.sh
```
# Testing with Pretrained Models:

```
sh test.sh
```





# bibtex
```
@article{zhao2023breathing,
  title={Breathing Life into Faces: Speech-driven 3D Facial Animation with Natural Head Pose and Detailed Shape},
  author={Zhao, Wei and Wang, Yijun and He, Tianyu and Yin, Lianying and Lin, Jianxin and Jin, Xin},
  journal={arXiv preprint arXiv:2310.20240},
  year={2023}
}
```

