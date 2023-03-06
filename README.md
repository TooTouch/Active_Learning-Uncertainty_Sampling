# Active Learning - Uncertainty Sampling

Pytorch implementation of "[A New Active Labeling Method for Deep Learning. IJCNN 2014](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6889457)".

# Environments

NVIDIA pytorch docker [ [link](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-12.html#rel-22-12) ]

```bash
docker pull nvcr.io/nvidia/pytorch:22.12-py3
```

`requirements.txt`

```bash
accelerate
wandb
torchvision
```


# Methods

`./query_strategies`

- Least Confidence
- Margin Sampling
- Entropy

<p align='center'>
    <img src="https://user-images.githubusercontent.com/37654013/222963947-df2071c4-ccd7-4a2c-8eb3-336f0dd72723.png">
</p>

# Results

**CIFAR10**

<p align='center'>
    <img src="https://github.com/TooTouch/Active_Learning-Uncertainty_Sampling/blob/main/results/CIFAR10/figure.jpg?raw=true">
</p>


# Reference

- cure-lab/deep-active-learning [ [link](https://github.com/cure-lab/deep-active-learning) ]
- Active Learning. Yi Zhang (CMU) [ [link](https://www.cs.cmu.edu/~tom/10701_sp11/recitations/Recitation_13.pdf) ]