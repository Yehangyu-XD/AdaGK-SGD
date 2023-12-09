import os, time, torch, torchvision
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from torchvision import datasets, transforms
from filelock import FileLock
import torch.multiprocessing as mp
from torchvision.datasets import CIFAR10, CIFAR100,MNIST,SVHN


# Dataloader
def get_dataloader(img_size, dataset, datapath, batch_size, seed ,no_val,):
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    WORKS = 0
    transform = transforms.Compose(
                    [
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                    )
    if img_size == 28:
        transform = transforms.Compose(
                    [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                    )
        img_size = 32
                    

    if img_size == 32:

        if dataset in ["svhn"]:
            train_set = datasets.SVHN(root=datapath,
                    split='train',
                    transform=transform,
                    download=True)
            val_set = datasets.SVHN(root=datapath,
                    split='test',
                    transform=transform,
                    download=True)
        elif dataset == 'MNIST':
            train_set = datasets.MNIST(root=datapath, 
						transform=transform,
						train=True,
						download=True
                          )
            val_set = datasets.MNIST(root=datapath, 
						transform=transform, 
						train=False)


        else:
            train_set = eval(dataset)(datapath, True, torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(img_size, padding=4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ]), download=True)
            val_set = eval(dataset)(datapath, True, torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    normalize,
                ]), download=True)

        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(0.1 * num_train))

        np.random.seed(seed)

        train_idx, valid_idx = indices[split:], indices[:split]                                             
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        test_set = eval(dataset)(datapath, False, torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                normalize,
            ]), download=True)

        distributed_train_sampler = DistributedSampler(
            train_set, 
            num_replicas=dist.get_world_size(), 
            rank = dist.get_rank(), 
            shuffle=False,
            )

        distributed_val_sampler = DistributedSampler(
            val_set,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
        )

        distributed_test_sampler = DistributedSampler(
            test_set,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False
        )

        if no_val:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, 
                sampler=distributed_train_sampler,
                num_workers=WORKS, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, sampler=distributed_val_sampler,
                num_workers=WORKS, pin_memory=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, sampler=distributed_train_sampler,
                num_workers=WORKS, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, sampler=distributed_val_sampler,
                num_workers=WORKS, pin_memory=True
            )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            num_workers=WORKS, pin_memory=False
        )
    else:
        raise ValueError("img_size must be 32")
    return train_loader, val_loader, test_loader, distributed_train_sampler



# training data 
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(), 
])

# test data 
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    

    def __len__(self):
        return len(self.x)


    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X,Y
        else:
            return X
