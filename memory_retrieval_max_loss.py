import argparse
import pandas as pd
import wandb
import numpy as np

from utils import *
from functions import *
from data import *
from utils.entmax import entmax15

parser = argparse.ArgumentParser()
parser.add_argument('--memory_size', type=int, default=100)
parser.add_argument('--data', type=str, default='mnist')
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--update_steps', type=int, default=1)
parser.add_argument('--kernel_epoch', type=int, default=100)
parser.add_argument('--activation', type=str, default='softmax')
parser.add_argument('--mode', type=str, default='MHN')
parser.add_argument('--kernel', type=str, default='lin')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--rerun', type=int, default=1)
parser.add_argument('--lr', type=float, default=10)

args = parser.parse_args()

torch.manual_seed(args.seed)

ACT_NAME = {
    'softmax':F.softmax,
    'sparsemax': sparsemax,
    'top20':topk_20,
    'top50':topk_50,
    'top80':topk_80,
    'random20':random_mask_02,
    'random50':random_mask_05,
    'random80':random_mask_08,
    'entmax':entmax15,
    'softmax1':softmax_1,
    'poly-10':polynomial
}

def sqdiff(x, y):
    x = torch.clamp(x, 0, 1)
    y = torch.clamp(y, 0, 1)
    sqdiff = torch.sum(torch.square(x - y), dim=-1)
    return torch.abs(sqdiff)

def memory_retrieval(Xi, update_rule, activation=F.softmax, overlap=dot_product, steps=1, beta=1):

    dist = []
    Xi = Xi.T
    for m in range(Xi.size(-1)):

        x = Xi[:, m].clone()
        q = torch.dropout(x, p=0.5, train=True)
        x_new = update_rule(Xi, q, beta, steps, overlap=overlap, activation=activation)
        dist.append(sqdiff(x, x_new).cpu().item())

    return np.mean(dist)

def main():

    m_size = args.memory_size
    if args.data == 'mnist':
        trainset, _ = load_mnist(m_size)
    elif args.data == 'cifar10':
        trainset, _ = load_cifar10(m_size)
    elif args.data == 'tiny_imagenet':
        trainset, _ = load_tiny_imagenet(m_size)
    elif args.data == 'synthetic':
        trainset = load_synthetic(m_size)

    Xi, _ = trainset[0]
    Xi = Xi.reshape(m_size, -1).cuda()

    if args.activation == 'softmax':
        activation = F.softmax
    elif args.activation == 'sparsemax':
        activation = sparsemax
    elif args.activation == 'poly-10':
        activation = polynomial
    elif args.activation == 'entmax':
        activation = entmax15
    else:
        activation = ACT_NAME[args.activation]

    if args.mode == 'MHN':
        overlap = dot_product
        update_rule = MHN_update_rule
        unif_loss = 100
    elif args.mode == 'UMHN':
        kernel, unif_loss = train_kernel_max(Xi[:int(0.5*m_size)], args.kernel_epoch, args.kernel)
        overlap = kernel.kernel_fn
        update_rule = UMHN_update_rule

    elif args.mode == 'Man':
        overlap = manhhatan_distance
        update_rule = MHN_update_rule
        unif_loss = 100
        
    elif args.mode == 'L2':
        overlap = l2_distance
        update_rule = MHN_update_rule
        unif_loss = 100

    init_unif = uniform_loss(Xi.T)

    config = vars(args)
    wandb.init(
        project="LMHN-Max",
        config=config
    )
    error = memory_retrieval(Xi, update_rule, activation, overlap, steps=args.update_steps, beta=args.beta)
    wandb.log({
        'error':error,
        'init_unif_loss':init_unif.item(),
        'unif_loss':unif_loss
    })
    wandb.finish()

main()