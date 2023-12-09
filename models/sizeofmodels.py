import math
from resnet_cifar import CifarResNet, ResNetBasicblock
from resnet_cifar import resnet20, resnet32, resnet44, resnet56, resnet110

# cifar based vggnet
from vgg_cifar import VGG
from vgg_cifar import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from model import googlenet

from model import densenet40
from torchvision.models import resnet18
from torchvision.models import mobilenet_v2
#from torchvision.models import alexnet,googlenet
net =densenet40(10)
total = sum(math.ceil(p.numel()) for p in net.parameters())
print('Total params:{}, as {} MB'.format(total,total*4/1024/1024))
