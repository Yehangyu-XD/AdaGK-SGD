python -m torch.distributed.launch --nproc_per_node=4 --nnode=1 --node_rank=0 \
--master_addr='127.0.0.1' --master_port=23333 \
main.py  \
--manualSeed=1111 \
--model_arch resnet20 --epochs 100 \
--batch_size 16 \
--dataset CIFAR10 \
--dataset_path ./data/cifar-10 \
--slowmo 0 \
--p_sync 1 \
--BMUF_Adam 0 \
--GlobalSynchronizationPeriod 20 \
--open_MLGK 1 \
--EASGD 0 ;
