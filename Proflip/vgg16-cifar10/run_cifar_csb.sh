#!/usr/bin/env 

DATE=`date +%Y-%m-%d`

############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=vgg16_quan
batch_size=128

#### TO CHANGE ####
dataset=finetune_mnist
program=CSB_nonadaptive
save_path=./results/${dataset}/$1
#### TO CHANGE ####

tb_path=${save_path}/tb_log  #tensorboard log path

PYTHON="python3 -m"
data_path='data'
    
echo $PYTHON

############### Neural network ############################
{
srun -p csc413 --gres gpu \
$PYTHON ${program} --dataset ${dataset} --data_path ${data_path} \
    --model ${model} --save_path ${save_path} \
    --batch_size ${batch_size} \
    --num_workers 1 \
    --chk_path $1/cifar60/model_best.pth.tar
    # --ic_only #default false
    #--bfa_mydefense
    # --clustering --lambda_coeff 1e-3    
} &
############## Tensorboard logging ##########################
{
if [ "$enable_tb_display" = true ]; then 
    sleep 30 
    wait
    $TENSORBOARD --logdir $tb_path  --port=6006
fi
} &
{
if [ "$enable_tb_display" = true ]; then
    sleep 45
    wait
    case $HOST in
    "Hydrogen")
        firefox http://0.0.0.0:6006/
        ;;
    "alpha")
        google-chrome http://0.0.0.0:6006/
        ;;
    esac
fi 
} &
wait