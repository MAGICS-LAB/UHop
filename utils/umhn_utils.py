import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class Kernel(nn.Module):
    def __init__(self, d):
        super(Kernel, self).__init__()
        self.w = nn.parameter.Parameter(torch.randn(d, d))

    def forward(self, x):
        # x: (D, n)
        return self.w @ x
    
    def kernel_fn(self, u, v):
        return (self.w@u).T @ (self.w@v)

class ICNN(nn.Module):
    def __init__(self, d):
        super(ICNN, self).__init__()
        self.w1 = nn.Linear(d, 200, bias=False)
        self.act = nn.ReLU()
        self.w2 = nn.Linear(d, 200)

        self.w3 = nn.Linear(200, d, bias=False)
        self.w4 = nn.Linear(d, d)

    def forward(self, x):
        # x: (D, n)
        x = x.T
        x1 = self.act(self.w1(x) + self.w2(x))
        out = self.w3(x1) + self.w4(x)
        return out
    
    def kernel_fn(self, u, v):
        return (self.forward(u)).T @ (self.forward(v))

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def uniform_loss_max(x, t=2):
    all_loss = torch.pdist(x, p=2).pow(2).mul(-t).exp().log()
    val, idx = torch.topk(  all_loss, k=1 )
    return all_loss[idx]

def separation_loss(x):
    sim = x @ x.T - torch.eye(x.size(0), device=x.device)
    loss = torch.triu(sim, diagonal=1).sum()
    return 1 - (loss/x.size(0))

def train_kernel(memory, epoch, kernel):
    # print(memory.size())
    if kernel == 'lin':
        k = Kernel(memory.size(-1)).cuda()
    elif kernel == 'icnn':
        k = ICNN(memory.size(-1)).cuda()

    # k= nn.DataParallel(k)

    opt = torch.optim.SGD(k.parameters(), lr=1)
    memory = memory.cuda()

    for i in range(epoch):
        opt.zero_grad()
        out = k(memory.T)

        loss = uniform_loss(F.normalize(out.T, dim=-1))
        loss.backward()
        opt.step()
        if i % 10 == 0:
          print( 'unif loss', round(loss.item(), 4))

        for g in opt.param_groups:
            g['lr'] = g['lr']/(math.sqrt(i+1))

    return k, loss.item()

def train_kernel_max(memory, epoch, kernel):
    # print(memory.size())
    if kernel == 'lin':
        k = Kernel(memory.size(-1)).cuda()
    elif kernel == 'icnn':
        k = ICNN(memory.size(-1)).cuda()

    # k= nn.DataParallel(k)

    opt = torch.optim.SGD(k.parameters(), lr=10)
    memory = memory.cuda()

    for i in range(epoch):
        opt.zero_grad()
        out = k(memory.T)

        loss = uniform_loss_max(F.normalize(out.T, dim=-1))
        loss.backward()
        opt.step()
        if i % 10 == 0:
          print( 'unif loss', round(loss.item(), 4))

        # for g in opt.param_groups:
        #     g['lr'] = g['lr']/(math.sqrt(i+1))

    return k, loss.item()