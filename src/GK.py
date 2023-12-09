############################################################################################################
#AdaGK-SGD: Adaptive Global Knowledge Guided Distributed Stochastic Gradient Descent
############################################################################################################
import torch
import torch.distributed as dist
from .utils import *
import copy

def average_weights(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size  

def average_something(something):
    size = float(dist.get_world_size())
    dist.all_reduce(something, op=dist.ReduceOp.SUM)
    something /= size 
    return something



def p_sync(model,global_weight,enable_p_sync,device_to_use):
    if enable_p_sync == 1:
        return model
    count = 0
    alpha = 2/(2+dist.get_world_size())
    for param in model.parameters():       
        temp = param.data 
        ii = (1-alpha)*global_weight[count]+alpha * temp
        temp.copy_(ii)
        count += 1
    return model


#K. Chen, H. Ding, and Q. Huo, “Parallelizing adam optimizer with blockwise model-update filtering,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2020.
def BMUF_Adam(model,optimizer,epoch_curr,global_weight,enable_BMUF_Adam,global_m_BMUF,global_m_BMUF_init,global_v_BMUF,global_v_BMUF_init,rho_n,tau,beta,k,device_to_use):
    if enable_BMUF_Adam == 1 :
        return model,rho_n,k,global_m_BMUF_init,global_v_BMUF_init,global_m_BMUF,global_v_BMUF
    count = 0
    eta = 1- 1/(dist.get_world_size())
    if epoch_curr % tau == 0:
        rho_n = eta * rho_n + tau 
    for param in model.parameters():  
        temp = param.data
        delar_n = temp - global_weight[count]
        ii =  temp +  eta * delar_n
        temp.copy_(ii)
        if epoch_curr % tau == 0:
            global_m_BMUF_init[count] = global_m_BMUF_init[count] *(beta[0]**(tau))*(beta[0]**(eta*rho_n)-1)/(1-beta[0]**(tau)) + optimizer.state_dict()['state'][count]['exp_avg']*(1-beta[0]**(tau+eta*rho_n))/(1-beta[0]**(tau))
            global_v_BMUF_init[count] = global_v_BMUF_init[count] * (beta[1]**(tau))*(beta[1]**(eta*rho_n)-1)/(1-beta[1]**(tau)) + optimizer.state_dict()['state'][count]['exp_avg_sq']*(1-beta[1]**(tau+eta*rho_n))/(1-beta[1]**(tau))
            optimizer.state_dict()['state'][count]['exp_avg'] = global_m_BMUF_init[count]
            optimizer.state_dict()['state'][count]['exp_avg_sq'] = global_v_BMUF_init[count]
            optimizer.state_dict()['state'][count]['step'] = optimizer.state_dict()['state'][count]['step']  + eta * rho_n
        count += 1
    
    
    return model,rho_n,k,global_m_BMUF_init,global_v_BMUF_init,global_m_BMUF,global_v_BMUF

#J. Wang, V. Tantia, N. Ballas, and M. Rabbat, “Slowmo: Improving communication-efficient distributed sgd with slow momentum,” arXiv preprint arXiv:1910.00643, 2019.
def slowmo(model,global_weight,u,optimizer,enable,device_to_use):
    if enable == 1:
        return model
    count = 0
    beta = 0.2
    alpha = 1
    for param in model.parameters():
        temp = param.data #.view(-1)
        u[count] = beta * u[count]+ (global_weight[count] - temp)/optimizer.state_dict()['param_groups'][0]['lr']
        ii = global_weight[count] - alpha *optimizer.state_dict()['param_groups'][0]['lr'] * u[count]
        temp.copy_(ii)
        count += 1
    return model

#S. Zhang, A. E. Choromanska, and Y. LeCun, “Deep learning with elastic averaging sgd,” in Proceedings of Advances in Neural Information Processing Systems (NeurIPS), vol. 28, 2015.
def EASGD(model,global_weight,u,enable,device_to_use):
    if enable == 1:
        return model
    count = 0
    alpha = 2/(dist.get_world_size()+2)
    beta = alpha
    
    for param in model.parameters():
        #print(type(param.data))
        temp = param.data #.view(-1)
        ii = temp - alpha*(global_weight[count]-u[count])
        temp.copy_(ii)
        u[count] = (1-beta)*u[count] + beta*global_weight[count]
        count += 1
        
    return model

#Maximum Lifetime of Global Knowledge
def MLGK(arg,model,optimizer,log,val_acc_ex,val_acc_aft,M_0,M_1,epoch_curr,GlobalSynchronizationPeriod, enable,best_prec1,best_epoch,method):
    if val_acc_aft >= val_acc_ex:
        is_best = val_acc_aft > best_prec1
        best_prec1 = max(val_acc_aft, best_prec1)
        if is_best:
            best_epoch = epoch_curr + 1
        if dist.get_rank() == 0:
            print_log(" acc best is: %.3f %%   at epoch %d " % (best_prec1,best_epoch),log)
        save_checkpoint({
            'epoch': epoch_curr + 1,
            'arch': arg.model_arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, os.path.join(arg.save_path, str(arg.dataset)+str(arg.model_arch)+method+'rank'+str(dist.get_rank())+'.pth'),
        os.path.join(arg.save_path, 'best'+str(arg.dataset)+str(arg.model_arch)+method+str(arg.manualSeed)+'rank'+str(dist.get_rank())+'.pth'))
                
    if epoch_curr % GlobalSynchronizationPeriod > M_0 and val_acc_aft <  M_1*val_acc_ex:
        enable = 1
        print_log('rank：{} \033[0;32m {} OFF at epoch:\033[0m {}'.format(dist.get_rank(),method,epoch_curr),log)
    
    return enable,best_prec1,best_epoch
    
