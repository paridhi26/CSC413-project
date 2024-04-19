# CSC413-project

# Instructions; adapted from the original README

## Installation

    python3 -m venv env
    source env/bin/activate
    python3 -m pip install -r requirements.txt


## Train Models
NOTE: It is recommended to run the training scripts on a machine with a GPU. The training scripts may take a long time to run otherwise. Moreover, the scripts below are written according to the authors' file structure. Upon downloading the code, the user may need to modify the shell variables in the scripts to match their own environment.

For each dataset and model, first train the base model, then the enhanced model.

### CIFAR10-resnet32
    cd cifar10/resnet32
    mkdir data
    sh train_CIFAR.sh
    sh finetune_CIFAR.sh
    
### CIFAR10-vgg16
    cd cifar10/vgg16
    mkdir data
    sh train_CIFAR.sh
    sh finetune_CIFAR.sh
    
A similar process is followed for MNIST.

## Evaluation
Evluation of the models may be run using 
    
    sh eval_ft_CIFAR.sh
    or sh eval_base_CIFAR.sh

These scripts may be easily modified for other models and datasets.

    
## ProFlip Attacks
First enter a folder to attack the target model, e.g. resnet-cifar10.

### Run non-adaptive attacks
    Run to generate a trigger: 
    sh run_cifar_trigger.sh {model-directory}
    Then attack:
    sh run_cifar_csb.sh {model-directory}
    
Similarly for MNIST.

### Results
The output includes the ASR, exit count, and post-attack validation accuracy. See Results section of the report for more details.
