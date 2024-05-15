# TSOANet

This repository provides the official PyTorch implementation of the following paper:

[TSOANet: Time-Sensitive Orthogonal Attention Network for medical event prediction](https://www.sciencedirect.com/science/article/pii/S0933365724001271)

    Chen, Hao, et al. "TSOANet: Time-Sensitive Orthogonal Attention Network for medical event prediction." Artificial Intelligence in Medicine (2024): 102885.

## Dataset
You can download datasets from the following links seperately:
    
1. [MIMIC-III](https://mimic.mit.edu/)

2. [eICU](https://eicu-crd.mit.edu/)

3. [MIMIC-III from DAPSNet](https://github.com/andylun96/DAPSNet)


## Requirements

    python==3.7
    pytorch==1.10.2
    pandas==1.3.5
    numpy==1.21.2

## Usage
To train a TSOANet model with dataset from DAPSNet, you can easily run the following command:

    bash train.sh

## Citiation
    @article{chen2024tsoanet,
    title={TSOANet: Time-Sensitive Orthogonal Attention Network for medical event prediction},
    author={Chen, Hao and Zhang, Junjie and Xiang, Yang and Lu, Shengye and Tang, Buzhou},
    journal={Artificial Intelligence in Medicine},
    pages={102885},
    year={2024},
    publisher={Elsevier}
    }
