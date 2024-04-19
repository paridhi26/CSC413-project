#!/usr/bin/env 

model=resnet32_quan
dataset='mnist' # trained on CIFAR, fine-tuned with MNIST

PYTHON="python3 -m"
data_path='./data'
chk_path=./saved_models/mnist30/model_best.pth.tar
save_path=./save_adversarial

 srun -p csc413 --gres gpu \
 $PYTHON adv_ex --dataset ${dataset} \
     --data_path ${data_path} --arch ${model} \
     --chk_path ${chk_path} --save_path ${save_path} \
     --seed 42