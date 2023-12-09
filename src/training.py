import os
import torch
import torch.distributed as dist
import time
from .utils import *
from .utils import (Average_and_new, secs2hours_mins_secs, print_log, time_string)
from .validate import validate
from .metrics import *
from .GK import *
from option import get_args

#global parameter
def set_global_parm() -> None:
    
    global arg_global
    arg_global = get_args()
    global device_to_use
    device_to_use = 'cuda:{}'.format(dist.get_rank())

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

    global global_weight_BMUF
    global_weight_easgd = []

    global enable_p_sync
    enable_p_sync= 1

    global enable_EASGD
    enable_EASGD = 1
    global enable_slowmo
    enable_slowmo = 1
    global enable_BMUF
    enable_BMUF = 0
    global k
    k = 0    
    global local_m_BMUF
    local_m_BMUF = []
    global local_v_BMUF
    local_v_BMUF = []
    global global_m_BMUF
    local_m_BMUF = []
    global global_v_BMUF
    local_v_BMUF = []
    global global_m_BMUF_init
    local_m_BMUF_init = []
    global global_v_BMUF_init
    local_v_BMUF_init = []
    global rho_n 
    rho_n = 0




#Training within a single epoch
def train_epoch(model, epoch_curr, train_sampler, train_loader, arg, criterion, optimizer, log, best,weight_average,tt,device_to_use, *args, **kwargs):
    batch_time = Average_and_new()
    data_time = Average_and_new()
    losses = Average_and_new()
    top1 = Average_and_new()
    top5 = Average_and_new()


       
    global global_weight
    global global_weight_slowmo
    global global_weight_easgd
    global global_weight_BMUF
    global local_m_BMUF
    global local_v_BMUF
    global k
    global GlobalMomentum
    global z_easgd
    global enable_p_sync
    global global_m_BMUF
    global global_v_BMUF
    global rho_n
    global global_m_BMUF_init
    global global_v_BMUF_init

    if arg.BMUF_Adam == 1:
        if epoch_curr == 0:
            k = 0
        e = 0.000001
        beta = (0.9,0.999)
        k = k + 1    

    if epoch_curr == 0 and arg.slowmo == 1:
        global_weight_slowmo = []
        GlobalMomentum = []
        for param in model.parameters():
            global_weight_slowmo.append(param.clone())
            GlobalMomentum.append(torch.zeros(param.shape).to(device_to_use))
    if epoch_curr == 0 and arg.EASGD == 1:
        global_weight_easgd = []
        z_easgd = []
        for param in model.parameters():
            global_weight_easgd.append(param.clone())
            z_easgd.append(torch.zeros(param.shape).to(device_to_use))
    
    if epoch_curr == 0 and arg.BMUF_Adam == 1:
        global_weight_BMUF = []
        local_m_BMUF = []
        local_v_BMUF = []
        global_m_BMUF = []
        global_v_BMUF = []
        for param in model.parameters():
            temp_global_m_BMUF = torch.zeros(param.shape,).to(device_to_use)
            local_m_BMUF.append(temp_global_m_BMUF.clone())
            local_v_BMUF.append(temp_global_m_BMUF.clone())
            global_weight_BMUF.append(param.clone())
            global_m_BMUF_init = copy.deepcopy(local_m_BMUF)
            global_v_BMUF_init = copy.deepcopy(local_m_BMUF)
            global_m_BMUF = copy.deepcopy(local_m_BMUF)
            global_v_BMUF = copy.deepcopy(local_m_BMUF)





    model.train()

    endtime = time.time()

    
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - endtime)
        target = target.to(device_to_use)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        if arg.use_cuda:
            input_var, target_var, model = input_var.to(device_to_use), target_var.to(device_to_use), model.to(device_to_use)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        batch_time.update(time.time() - endtime)
        endtime = time.time()
        if i % arg.print_freq == 0 and dist.get_rank() == 0 and epoch_curr < 500:
            print_log('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch_curr + 1, i, len(train_loader),
                                                                      batch_time=batch_time,
                                                                      data_time=data_time,
                                                                      loss=losses,
                                                                      top1=top1,
                                                                      top5=top5), log)
    


#train
def train(arg, model, optimizer, train_sampler, train_loader, criterion, val_loader, log, scheduler,device_to_use,method,beta,*args, **kwargs,):
    set_global_parm() 
    start_time = time.time()
    epoch_time = Average_and_new()
    best_prec1 = 0
    best_epoch = 0
    global Accuracy_history
    global global_weight
    GlobalSynchronizationPeriod = arg.GlobalSynchronizationPeriod
    weight_average = 0
    global global_weight
    global global_weight_slowmo
    global global_weight_easgd
    global GlobalMomentum
    global z_easgd
    global enable_p_sync
    global enable_EASGD
    global enable_slowmo
    global enable_BMUF
    global global_m_BMUF
    global global_v_BMUF
    global global_weight_BMUF
    global global_m_BMUF_init
    global global_v_BMUF_init
    global rho_n
    global k
    global local_m_BMUF
    global local_v_BMUF
    

    if arg.EASGD == 1:
        enable_p_sync = 1
        enable_slowmo = 1
    
    if arg.open_MLGK == 1:
        M_0 = 0
        M_1 = 1
    else:
        M_0 = -1
        M_1 = 2



    for epoch_curr in range(arg.start_epoch, arg.epochs):

        need_hour, need_mins, need_secs = secs2hours_mins_secs(epoch_time.val * (arg.epochs - epoch_curr))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        if dist.get_rank() == 0:
            print_log(' [{:s}] :: {:3d}/{:3d} ----- [{:s}] {:s}'.format(arg.model_arch, epoch_curr + 1, arg.epochs,
                                                                        time_string(), need_time), log)
       
                

        train_epoch(model, epoch_curr,train_sampler, train_loader, arg, criterion, optimizer, log,best_epoch,weight_average,GlobalSynchronizationPeriod,device_to_use)
        
        if weight_average == 0 and epoch_curr % GlobalSynchronizationPeriod == 0 :

            if arg.p_sync == 1:
                curr_weight = []
                for param in model.parameters():
                    curr_weight.append(param.clone())

        #EASGD       
        if arg.EASGD == 1:
            if enable_EASGD == 0:
                val_acc_0 = validate(model, val_loader, criterion, arg.print_freq, log,device_to_use)

            model =EASGD(model,global_weight_easgd,z_easgd,enable_EASGD,device_to_use)

            if enable_EASGD == 0:
                val_acc_EASGD = validate(model, val_loader, criterion, arg.print_freq, log,device_to_use)
                enable_EASGD,best_prec1,best_epoch = MLGK(arg, model, optimizer, log, val_acc_0, val_acc_EASGD, M_0, M_1, epoch_curr, GlobalSynchronizationPeriod,enable_EASGD, best_prec1, best_epoch, method)  


        if weight_average == 0 and epoch_curr % GlobalSynchronizationPeriod == 0 :
            print_log('\033[0;31m do average weights! at epoch:\033[0m {}'.format(epoch_curr),log)
            average_weights(model)
            if arg.BMUF_Adam == 1 and enable_BMUF == 0: 
                count = 0
                for param in model.parameters():
                    optimizer.state_dict()['state'][count]['exp_avg'] = average_something(optimizer.state_dict()['state'][count]['exp_avg'])
                    optimizer.state_dict()['state'][count]['exp_avg_sq'] = average_something(optimizer.state_dict()['state'][count]['exp_avg_sq'])
                    count += 1

            if arg.EASGD == 1:
                enable_EASGD = 0   
                print_log('\033[0;34m {} ON at epoch:\033[0m {}'.format(method,epoch_curr),log)
            
            if arg.slowmo == 1:
                enable_slowmo = 0   
                print_log('\033[0;34m {} ON at epoch:\033[0m {}'.format(method,epoch_curr),log)
        
        
            if arg.p_sync == 1:
                enable_p_sync = 0
                print_log('\033[0;34m {} ON at epoch:\033[0m {}'.format(method,epoch_curr),log)
                global_weight = []
                for param in model.parameters():
                    global_weight.append(param.clone())

                for w in range(len(global_weight)):
                    global_weight[w] -=  curr_weight[w]/ float(dist.get_world_size())
                    global_weight[w] = global_weight[w]*(float(dist.get_world_size())/(float(dist.get_world_size())-1))
            
            if arg.BMUF_Adam == 1:
                enable_BMUF = 0
                print_log('\033[0;34m {} ON at epoch:\033[0m {}'.format(method,epoch_curr),log)
        
        #BMUF_Adam
        if arg.BMUF_Adam == 1:
            val_acc_0 = validate(model, val_loader, criterion, arg.print_freq, log,device_to_use)
            model,rho_n,k,global_m_BMUF_init,global_v_BMUF_init,global_m_BMUF,global_v_BMUF = BMUF_Adam(model,optimizer,epoch_curr,global_weight_BMUF,enable_BMUF,global_m_BMUF,global_m_BMUF_init,global_v_BMUF,global_v_BMUF_init,rho_n,GlobalSynchronizationPeriod,beta,k,device_to_use)
            if enable_BMUF == 0:
                val_acc_BMUF_Adam = validate(model, val_loader, criterion, arg.print_freq, log,device_to_use)
                enable_BMUF,best_prec1,best_epoch = MLGK(arg, model, optimizer, log, val_acc_0, val_acc_BMUF_Adam, M_0, M_1,epoch_curr, GlobalSynchronizationPeriod,enable_BMUF,best_prec1, best_epoch, method)
                

        #slowmo
        if arg.slowmo == 1:
            val_acc_0 = validate(model, val_loader, criterion, arg.print_freq, log,device_to_use)
            model = slowmo(model,global_weight_slowmo,GlobalMomentum,optimizer,enable_slowmo,device_to_use)
  
            
            if enable_slowmo == 0: 
                val_acc_slowmo = validate(model, val_loader, criterion, arg.print_freq, log,device_to_use)
                enable_slowmo,best_prec1,best_epoch = MLGK(arg, model, optimizer, log, val_acc_0, val_acc_slowmo, M_0, M_1, epoch_curr, GlobalSynchronizationPeriod,enable_EASGD, best_prec1, best_epoch, method)
        
        if weight_average == 0 and epoch_curr % GlobalSynchronizationPeriod == 0 :
            if arg.slowmo == 1:
                global_weight_slowmo = []
                for param in model.parameters():
                    global_weight_slowmo.append(param.clone())

        
            if arg.EASGD == 1:     
                global_weight_easgd = []
                for param in model.parameters():
                    global_weight_easgd.append(param.clone())
            
            if arg.BMUF_Adam == 1:
                global_weight_BMUF = []
                for param in model.parameters():
                    global_weight_BMUF.append(param.clone())

        val_acc_2 = validate(model, val_loader, criterion, arg.print_freq, log,device_to_use)


        scheduler.step()
        Accuracy_history.append(val_acc_2.item())
        
        # remember best prec@1 and save checkpoint
        is_best = val_acc_2 > best_prec1
        best_prec1 = max(val_acc_2, best_prec1)
        if is_best:
            best_epoch = epoch_curr + 1
        if dist.get_rank() == 0:
            print_log(" acc best is: %.3f %%   at epoch %d" % (best_prec1,best_epoch),log)
        save_checkpoint({
            'epoch': epoch_curr + 1,
            'arch': arg.model_arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
         }, is_best, os.path.join(arg.save_path, str(arg.dataset)+str(arg.model_arch)+method+'rank'+str(dist.get_rank())+'.pth'),
        os.path.join(arg.save_path, 'best'+str(arg.dataset)+str(arg.model_arch)+method+str(arg.manualSeed)+'rank'+str(dist.get_rank())+'.pth'))
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        if arg.p_sync == 1:
            model = p_sync(model,global_weight,enable_p_sync,device_to_use)
        if enable_p_sync == 0:
            val_acc_3 = validate(model, val_loader, criterion, arg.print_freq, log,device_to_use)
            enable_p_sync,best_prec1,best_epoch = MLGK(arg, model, optimizer, log, val_acc_2, val_acc_3, M_0, M_1, epoch_curr, GlobalSynchronizationPeriod,enable_p_sync, best_prec1, best_epoch, method)

    if dist.get_rank() == 0:
        data_write('result/Accuracy_history/Accuracy_history_{}_{}_{}_{}.xlsx'.format(arg.dataset,arg.model_arch,arg.manualSeed,method),Accuracy_history)
   