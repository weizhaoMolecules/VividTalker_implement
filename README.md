#  Breathing Life into Faces: Speech-driven 3D Facial Animation with Natural Head Pose and Detailed Shape
This repository contains a pytorch implementation of "Breathing Life into Faces: Speech-driven 3D Facial Animation with Natural Head Pose and Detailed Shape"


> <a href='https://arxiv.org/abs/2310.20240'><img src='https://img.shields.io/badge/arXiv-2301.02379-red'></a> <a href='https://weizhaomolecules.github.io/VividTalker/'><img src='https://img.shields.io/badge/Project-Video-Green'></a>

![](src/data/teaser.gif)

This codebase provides:
- train code
- test code
- dataset
- checkpoints

<p align="center">
<img src="architecture.png" width="75%"/>
</p>

## **Environment**
- Linux
- Python 3.7+
- Pytorch 1.10.0
- CUDA 11.7 

Other necessary packages:
```
pip install -r requirements.txt
```


## Installation:
python 3.7.11

```
cd Vividtalker/src/
conda create -n vividtalker python=3.7
conda activate vividtalker
pip install -r requirements.txt
```

# 3D-VTFSET
https://drive.google.com/file/d/120EXgiqPrYP2ZjNq59ETwhbQdy_7wPmQ/view?usp=drive_link



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
# Testing:

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

