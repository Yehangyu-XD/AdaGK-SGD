############################################################################################################
#AdaGK-SGD: Adaptive Global Knowledge Guided Distributed Stochastic Gradient Descent
############################################################################################################

import models
import argparse

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))


def get_args() -> argparse.Namespace:
    """parser args"""
    parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset options
    #parser.add_argument('--data_dir', type=str, help='Path to dataset', default='./data/cifar-10')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset', default='./data/cifar-10')
    parser.add_argument('--dataset', type=str, metavar='NAME', choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'svhn', 'stl10','MNIST'],
                        help='Choose between CIFAR10/100 and ImageNet.', default='CIFAR10')

    # DDP input
    parser.add_argument('--local_rank', type=int, default= 0)
    parser.add_argument('--nproc_per_node', type=str, default='1')
    parser.add_argument('--nnode', type=str, default='1')
    parser.add_argument('--node_rank', type=str, default='0')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1')
    parser.add_argument('--master_port', type=str, default='23333')
    parser.add_argument('--slowmo', type=int, default=0) 
    parser.add_argument('--p_sync', type=int, default=0)
    parser.add_argument('--EASGD', type=int, default=0)
    parser.add_argument('--BMUF_Adam', type=int, default=0)
    parser.add_argument('--GlobalSynchronizationPeriod', type=int, default=20)
    parser.add_argument('--local_sync_prop', type=float, default=0.1)
    parser.add_argument('--open_MLGK', type=int, default=0)

    
    # The path of files to save
    parser.add_argument('--save_path', type=str, default='./result/', help='Folder to save checkpoints and log.')

    # Model options
    parser.add_argument('--model_arch', type=str, metavar='ARCH', default='resnet20', choices=model_names)

    # Optimization options
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='The Initial Learning Rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')#warmup_epoch
    parser.add_argument('--warmup_epochs', type=int, default=5, help='epochs for warmup.')
    

    # Checkpoints
    parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set', default=False)

    # Acceleration
    #parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--num_gpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')

    # Random seed
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--seed', type=str, default='1111')



    

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)