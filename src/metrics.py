############################################################################################################
#AdaGK-SGD: Adaptive Global Knowledge Guided Distributed Stochastic Gradient Descent
############################################################################################################
import torch.distributed as dist
from .utils import *


#Computational average accuracy
def average_acc(acc):
    size = float(dist.get_world_size())
    dist.all_reduce(acc, op=dist.ReduceOp.SUM)
    acc /= size

#Computational accuracy
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
