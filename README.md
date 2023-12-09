# Project for AdaGK-SGD

This project uses Pytorch to simulate AdaGK-SGD. (AdaGK-SGD: Adaptive Global Knowledge Guided Distributed Stochastic Gradient Descent)

## Dataset

Please place the data set in the /data path.

| Data sets that are validated for use： |
| -------------------------------------- |
| CIFAR-10                               |
| CIFAR100                               |
| ILSVRC2012                             |
| MNIST                                  |

## Models

| This project includes a variety of models. |
| ------------------------------------------ |
| VGG                                        |
| ResNet                                     |
| DenseNet                                   |
| MobileNet                                  |

## Other algorithms

In addition to AdaGK-SGD, the project also includes the implementation of a variety of comparison algorithms.

| Other algorithms in the project |                                                                                                                                                                                                               |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Local SGD                       | *B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Artificial Intelligence and Statistics (AISTATS), 2017.* |
| SlowMo                          | *J. Wang, V. Tantia, N. Ballas, and M. Rabbat, “Slowmo: Improving communication-efficient distributed sgd with slow momentum,” arXiv preprint arXiv:1910.00643, 2019.*                                    |
| EASGD                           | *S. Zhang, A. E. Choromanska, and Y. LeCun, “Deep learning with elastic averaging sgd,” in Proceedings of Advances in Neural Information Processing Systems (NeurIPS), vol. 28, 2015.*                    |
| BUMF-Adam                       | *K. Chen, H. Ding, and Q. Huo, “Parallelizing adam optimizer with blockwise model-update filtering,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2020.*         |

The MLGK module in this project can be opened simultaneously with the above comparison algorithm to improve the performance of the algorithm.

## Result

Experiment results and run logs are saved in /result. The results are sorted by seed. You can find the detailed model convergence process under /result/Accuracy_history.

## Running the project

You can run the step file directly to start the program

```shell
sh run.sh
```

| Input parameter description | slowmo | p_sync | BMUF_Adam | EASGD | MLGK |
| --------------------------- | :----: | :----: | :-------: | :---: | :--: |
| AdaGK-SGD:                  |   0   |   1   |     0     |   0   |  1  |
| SlowMo:                     |   1   |   0   |     0     |   0   |  0  |
| EASGD:                      |   0   |   0   |     0     |   1   |  0  |
| BUMF-Adam:                  |   0   |   0   |     1     |   0   |  0  |
| Local SGD:                  |   0   |   0   |     0     |   0   |  0  |
| MLGK-SlowMo:                |   1   |   0   |     0     |   0   |  1  |
| MLGK-EASGD:                 |   0   |   0   |     0     |   1   |  1  |
| MLGK-BUMF-Adam:             |   0   |   0   |     1     |   0   |  1  |

## Requirments

* matplotlib=3.6.0
* opencv-python=4.6.0.66
* openpyxl=3.0.10
* openssl=1.1.1q
* pandas=1.5.0
* python=3.8.13
* pytorch=1.11.0
* torchvision=0.12.0
* warmup-scheduler=0.3
