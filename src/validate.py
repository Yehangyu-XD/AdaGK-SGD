############################################################################################################
#AdaGK-SGD: Adaptive Global Knowledge Guided Distributed Stochastic Gradient Descent
############################################################################################################
import torch
import torch.distributed as dist
import time
from .utils import *
from data.datasets import get_dataloader
from .utils import (Average_and_new, best_loss_and_epoch, secs2hours_mins_secs, print_log, time_string)
import torch.backends.cudnn as cudnn
from .metrics import *

#validate
def validate(model, val_loader, criterion, print_freq, log,device_to_use):
    
    batch_time = Average_and_new()
    losses = Average_and_new()
    top1 = Average_and_new()
    top5 = Average_and_new()
    model.eval()

    start_timepoint = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device_to_use,non_blocking=True)

        with torch.no_grad():
            input_var = torch.autograd.Variable(input).to(device_to_use)
            target_var = torch.autograd.Variable(target).to(device_to_use)
        output = model(input_var)
        loss = criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        test_loss = loss.item()
   
        losses.update(test_loss, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        batch_time.update(time.time() - start_timepoint)
        start_timepoint = time.time()

        if i % print_freq == 0 and dist.get_rank() == 0:
            print_log('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5), log)
    if  dist.get_rank() == 0:
        print_log(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                               error1=100 - top1.avg),
                                                                                               log)

    return top1.avg