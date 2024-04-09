#!/usr/bin/env 

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"alpha")
    PYTHON="/usr/bin/python3.6" # python environment path
    TENSORBOARD='/home/elliot/anaconda3/envs/pytorch041/bin/tensorboard' # tensorboard environment path
    data_path='./data'
    ;;
esac

DATE=`date +%Y-%m-%d`



############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=resnet32_quan
dataset=finetune_mnist # trained on CIFAR, fine-tuned with MNIST
epochs=30
train_batch_size=128
test_batch_size=128
optimizer=Adam

label_info=binarized

attack_sample_size=128 # number of data used for BFA
n_iter=20 # number of iteration to perform BFA
k_top=10 # only check k_top weights with top gradient ranking in each layer

save_path=./save_tmp
tb_path=${save_path}/tb_log  #tensorboard log path

PYTHON="python3 -m"
data_path='./data'
pretrained_model=./save_finetune/cifar60/model_best.pth.tar

echo $PYTHON

############### Neural network ############################
{
srun -p csc413 --gres gpu \
$PYTHON main --dataset ${dataset} --data_path ${data_path}   \
    --arch ${model} --save_path ${save_path} \
    --epochs ${epochs} --learning_rate 0.0004 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --attack_sample_size ${train_batch_size} \
    --test_batch_size ${test_batch_size} \
    --workers 1 --ngpu 1 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --resume ${pretrained_model} \
    --ic_only True \
    --evaluate \
    --attack_sample_size ${attack_sample_size} \
    --reset_weight --bfa --n_iter ${n_iter} --k_top ${k_top} \

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