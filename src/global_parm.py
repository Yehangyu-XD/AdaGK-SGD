############################################################################################################
#AdaGK-SGD: Adaptive Global Knowledge Guided Distributed Stochastic Gradient Descent
############################################################################################################
import torch.distributed as dist
from utils import *
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

    global enable_p_sync
    enable_p_sync= 1

    global enable_EASGD
    enable_EASGD = 1
    global enable_slowmo
    enable_slowmo = 1
