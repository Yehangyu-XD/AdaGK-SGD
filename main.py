############################################################################################################
#AdaGK-SGD: Adaptive Global Knowledge Guided Distributed Stochastic Gradient Descent
############################################################################################################

import os
import shutil
import torch
import torch.distributed as dist
import time
from src.utils import *
import models
from data.datasets import get_dataloader
from src.utils import (Average_and_new, best_loss_and_epoch, secs2hours_mins_secs, print_log, time_string)
import torch.backends.cudnn as cudnn
import pandas as pd
from option import get_args
import copy
from warmup_scheduler import GradualWarmupScheduler
import datetime
from src.training import train
from src.validate import validate
from src.metrics import *
from src.GK import *


#global parameter
def set_global_parm() -> None:
    
    global arg_global
    arg_global = get_args()
    global device_to_use
    device_to_use = 'cuda:{}'.format(dist.get_rank())

    #Count the number of global synchronization times
    global num_aw
    num_aw = 0
    global num_ag
    num_ag = 0

    global Accuracy_history
    Accuracy_history = []

    global method
    method = 'NULL'

    #Cache temporary global weight
    global global_weight
    global_weight = []
    #Cache the temporary global weights under slowmo
    global global_weight_slowmo
    global_weight_slowmo = []

    global global_weight_easgd
    global_weight_easgd = []

    global enable_p_sync
    enable_p_sync= 1

    global enable_EASGD
    enable_EASGD = 1

    global enable_slowmo
    enable_slowmo = 1


#Save model
def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


#Output the weight to excel
def data_write(file_path, datas,columns=['acchistory']):

    df = pd.DataFrame(datas, columns=columns)
    df.to_excel(file_path, index=False)


   
def main():
    args = get_args()
    args.use_cuda = args.num_gpu > 0 and torch.cuda.is_available()
    set_seed(args.manualSeed, args.use_cuda)
    cudnn.benchmark = False
    cudnn.deterministic = True
    global method
    global device_to_use
    

    if args.p_sync == 1 :
        method = 'baseMLGK-SGD'
    elif args.EASGD ==1 :
        if args.open_MLGK == 1:
            method = 'MLGK-EASGD'
        else:
            method = 'EASGD'
    elif args.slowmo ==1:
        if args.open_MLGK == 1:
            method = 'MLGK-slowmo'
        else:
            method = 'slowmo'
    elif args.BMUF_Adam ==1:
        if args.open_MLGK == 1:
            method = 'MLGK-BMUF_Adam'
        else:
            method = 'BMUF_Adam'
    else :
        method = 'Local SGD'

    log = init_logger(args, method)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    worldsize = dist.get_world_size()

    

    print_log('==>> '+method+' <<==', log)



    # Data
    if dist.get_rank == 0:
        print_log('==>>> Preparing data..', log)
    if not os.path.isdir(args.dataset_path):
        os.makedirs(args.dataset_path)
    if args.dataset == 'CIFAR10':
        num_cls = 10
        img_size = 32
    elif args.dataset == 'CIFAR100':
        num_cls = 100
        img_size = 32
    elif args.dataset == 'svhn':
        num_cls = 10
        img_size = 32
    elif args.dataset == 'MNIST':
        num_cls = 10
        img_size = 28
        inplanes = 1
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)
    train_loader, val_loader, test_loader, train_sampler = get_dataloader(img_size, args.dataset, args.dataset_path, args.batch_size, args.manualSeed, no_val=True, )

    # Init model
    if dist.get_rank == 0:
        print_log("==>>> creating model '{}'".format(args.model_arch), log)
    model = models.__dict__[args.model_arch](num_classes=num_cls)
    if dist.get_rank() == 0:
        print_log("==>>> model :\n {}".format(model), log)

    beta = (0.9,0.99) #Adam hyperparameter

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if args.BMUF_Adam ==1:
        optimizer = torch.optim.Adam(model.parameters(), 1e-4, betas=beta, eps=1e-08,weight_decay=state['decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), worldsize*state['lr'], momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=0.000001, last_epoch=-1, verbose=False)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=args.warmup_epochs ,after_scheduler=scheduler)

    if args.use_cuda:
        model.to(device_to_use)
        criterion.to(device_to_use)
    recorder = best_loss_and_epoch(args.epochs)

    if args.resume:
        if os.path.isfile(args.resume):
            if dist.get_rank()==0:
                print_log("===>>> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            if args.use_state_dict:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model = checkpoint['state_dict']

            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("===>>> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("===>>> no checkpoint found at '{}'".format(args.resume), log)
    else:
        if dist.get_rank()==0:
            print_log("===>>> do not use any checkpoint for {} model".format(args.model_arch), log)


    if args.evaluate:
        time1 = time.time()
        validate(model, test_loader, criterion, args.print_freq,log,device_to_use)
        time2 = time.time()
        print('function took %0.3f ms' % ((time2 - time1) * 1000.0))
        return
    
    if dist.get_rank()==0:
        print("-" * 10 + "one epoch begin" + "-" * 10)
    average_weights(model)
    val_acc_1 = validate(model, test_loader, criterion, args.print_freq,log,device_to_use)
    average_acc(val_acc_1)
    if dist.get_rank == 0:
        print(" acc before is: %.3f %%" % val_acc_1)

    # NOTE train
    start_time = time.time()
    train(args, model, optimizer, train_sampler, train_loader, criterion, test_loader, log, scheduler,device_to_use,method,beta)   
    checkpoint = torch.load(args.save_path+'best'+str(args.dataset)+str(args.model_arch)+method+str(args.manualSeed)+'rank'+str(dist.get_rank())+'.pth')
    model.load_state_dict(checkpoint['state_dict'])
    val_acc_2 = validate(model, test_loader, criterion, args.print_freq,log,device_to_use)
    average_acc(val_acc_2)
    if  dist.get_rank() == 0:
        print_log("used time:{}".format(time.time()-start_time),log)
        print_log(" acc after is: %.3f %%" % val_acc_2,log)

    log.close()

if __name__ == "__main__":

    my_timeout = datetime.timedelta(days=0, seconds=72000, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        group_name='test',
        timeout=my_timeout
    )
    print('===>>> DDP init successfully <<<===')
    print('===>>> rank is {} <<<==='.format(dist.get_rank()))
    set_global_parm()
    print(device_to_use)
    main()



    
