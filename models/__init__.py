"""The models subpackage contains definitions for the following model
architectures:
-  `ResNeXt` for CIFAR10 CIFAR100
You can construct a model with random weights by calling its constructor:
.. code:: python
    import models
    resnext29_16_64 = models.ResNeXt29_16_64(num_classes)
    resnext29_8_64 = models.ResNeXt29_8_64(num_classes)
    resnet20 = models.ResNet20(num_classes)
    resnet32 = models.ResNet32(num_classes)


.. ResNext: https://arxiv.org/abs/1611.05431
"""

# 
import imp
from .classifier4test import Classifier
from .classifier4test import classifier

# cifar based resnet
from .resnet_cifar import CifarResNet, ResNetBasicblock
from .resnet_cifar import resnet20, resnet32, resnet44, resnet56, resnet110

# cifar based vggnet
from .vgg_cifar import VGG
from .vgg_cifar import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn

from .model import densenet40
from torchvision.models import resnet50
from torchvision.models import mobilenet_v2
from torchvision.models import alexnet
from .model import googlenet


