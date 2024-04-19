model=vgg16_quan
dataset='mnist' # if the model is trained on CIFAR, fine-tuned with MNIST, use "finetune_mnist", "mnist" otherwise
PYTHON="python3 -m"

data_path='./data'
chk_path=./save_aug/cifar60_ft/vgg_mm_wrob.tar # path of model, notice: cifar60_ft is for cifar and mnist, mnist directory is for trained only on mnist
save_path=./save_adversarial # this is the path where it used to store the result (adv_ex.txt). NOTICE: the result got overwritten everytime run the script

 srun -p csc413 --gres gpu \
 $PYTHON adv_ex --dataset ${dataset} \
     --data_path ${data_path} --arch ${model} \
     --chk_path ${chk_path} --save_path ${save_path} \
     --seed 42