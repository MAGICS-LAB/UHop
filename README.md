# Uniform Memory Retrieval with Larger Capacity for Modern Hopfield Models

This is the Official Code for the paper: **Uniform Memory Retrieval with Larger Capacity for Modern Hopfield Models**.

## Create Environment
` conda create -n uhop python=3.8` <br>
`conda activate uhop` <br>
`pip3 install -r requirements.txt` <br>

## Memory Retrieval Task

### For U-Hop:
`python3 memory_retrieval_max_loss.py --memory_size 100 --kernel_epoch 100 --activation "softmax" --data "cifar10" --mode "UMHN" --seed 42
`

### For Modern Hopfield:
`python3 memory_retrieval_max_loss.py --memory_size 100 --activation "softmax" --data "cifar10" --mode "MHN" --seed 42
`

### For Sparse Modern Hopfield:
`python3 memory_retrieval_max_loss.py --memory_size 100 --activation "sparsemax" --data "cifar10" --mode "MHN" --seed 42
`

## Noise Robustness Task

### For U-Hop:
`python3 memory_retrieval_noise.py --noise_level 0.5 --kernel_epoch 100 --activation "softmax" --data "cifar10" --mode "UMHN" --seed 42
`

### For Modern Hopfield:
`python3 memory_retrieval_noise.py --noise_level 0.5 --activation "softmax" --data "cifar10" --mode "MHN" --seed 42
`

### For Sparse Modern Hopfield:
`python3 memory_retrieval_noise.py --noise_level 0.5 --activation "sparsemax" --data "cifar10" --mode "MHN" --seed 42
`

## Image Classification on CIFAR10 and CIFAR100

`python3 image_classification.py --data cifar10 --datasize 10000 --n_class 10`


## Image Classification on Tiny ImageNet

To run experiments on TinyImageNet, you can use the code `download_tinyimagenet.sh` to download the dataset.
If you have downloaded the dataset already, please see `data_utils.py` to setup the corresponding directory.

`python3 deep_ViH.py --data tiny_imagenet --datasize 60000 --n_class 200 --init_lr 0.0001 --batch_size 1024`

## Citation

If you find our paper useful, please consider citing our work

```
@inproceedings{wu2024uniform,
  title={Uniform Memory Retrieval with Larger Capacity for Modern Hopfield Models},
  author={Wu, Dennis and Hu, Jerry Yao-Chieh and Hsiao, Teng-Yun and Liu, Han},
  booktitle={Forty-first International Conference on Machine Learning (ICML)},
  year={2024},
  url={https://arxiv.org/abs/2404.03827}
}
```
