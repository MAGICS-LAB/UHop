import argparse
import pandas as pd
import wandb
import numpy as np

from utils import *
from functions import *
from data import *
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--memory_size', type=int, default=100)
parser.add_argument('--data', type=str, default='mnist')
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--update_steps', type=int, default=1)
parser.add_argument('--kernel_epoch', type=int, default=25)
parser.add_argument('--activation', type=str, default='softmax')
parser.add_argument('--mode', type=str, default='MHN')
parser.add_argument('--kernel', type=str, default='lin')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

torch.manual_seed(args.seed)


def sqdiff(x, y):
    x = torch.clamp(x, 0, 1)
    y = torch.clamp(y, 0, 1)
    sqdiff = torch.sum(torch.square(x - y), dim=-1)
    return torch.abs(sqdiff)

def memory_calculation(Xi, overlap=dot_product):

    ratios = []
    Xi = Xi.T
    for m in range(Xi.size(-1)):

        x = Xi[:, m].clone()
        q = torch.dropout(x, p=0.5, train=True)
        ratio = compute_energy_contribution(Xi, q, m, overlap)
        ratios.append(ratio)

    return np.mean(ratios)

def compute_energy_contribution(memory, query, idx, overlap):
    # x: D, Xi: (D, M)
    e = -torch.logsumexp((overlap(memory, query).long()), dim=0) + 0.5*(torch.dot(query,query))
    part = -torch.logsumexp((overlap(memory[:, idx], query).long()), dim=0) + 0.5*(torch.dot(query,query))
    return (part/e).item()

def main():

    m_size = args.memory_size

    data = {
        'memory size':[],
        'model':[],
        'target memory energy contribution (%)':[]
    }
    
    for m_size in [1000, 5, 10, 50, 100, 200, 500, 1000]:

        for s in range(5):
            
            torch.manual_seed(s*20)

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


            ratio = memory_calculation(Xi, dot_product)
            data['memory size'].append(m_size)
            data['model'].append('Modern Hopfield')
            data['target memory energy contribution (%)'].append(ratio*100)

            kernel, _ = train_kernel(Xi, 1, args.kernel)
            ratio = memory_calculation(Xi, kernel.kernel_fn)
            data['memory size'].append(m_size)
            data['model'].append('Modern Hopfield + U-HOP (N=1)')
            data['target memory energy contribution (%)'].append(ratio*100)


            kernel, _ = train_kernel(Xi, 2, args.kernel)
            ratio = memory_calculation(Xi, kernel.kernel_fn)
            data['memory size'].append(m_size)
            data['model'].append('Modern Hopfield + U-HOP (N=2)')
            data['target memory energy contribution (%)'].append(ratio*100)


            kernel, _ = train_kernel(Xi, 5, args.kernel)
            ratio = memory_calculation(Xi, kernel.kernel_fn)
            data['memory size'].append(m_size)
            data['model'].append('Modern Hopfield + U-HOP (N=5)')
            data['target memory energy contribution (%)'].append(ratio*100)

            kernel, _ = train_kernel(Xi, 10, args.kernel)
            ratio = memory_calculation(Xi, kernel.kernel_fn)
            data['memory size'].append(m_size)
            data['model'].append('Modern Hopfield + U-HOP (N=10)')
            data['target memory energy contribution (%)'].append(ratio*100)


    plt.tight_layout()
    sns.lineplot(data=data, x='memory size', y='target memory energy contribution (%)', hue='model', marker="o")
    plt.savefig('energy_contribution.png', transparent=True)


main()