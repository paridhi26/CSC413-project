import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import models
import random
from models.quantization import quan_Conv2d, quan_Linear, quantize
from utils import AverageMeter, RecorderMeter
import torch.nn.functional as F
from collections import Counter
import os, sys
import argparse

# Constants
BATCH_SIZE = 1
use_cuda = True
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
NUM_CHANNELS = 3
NUM_CLASSES = 10
MEAN, STD = [0.5], [0.5]
weight='1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1'
w = [float(i) for i in weight.split(',')]


import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset', default='finetune_mnist', type=str, help='dataset')
    parser.add_argument('--data_path', default='./data', type=str, help='data path')
    parser.add_argument('--arch', default='resnet32_quan', type=str, help='architecture')
    parser.add_argument('--chk_path', default='./save_finetune/cifar60/model_best.pth.tar', type=str, help='checkpoint path')
    parser.add_argument('--save_path', default='./save_adversarial/', type=str, help='save path')
    parser.add_argument('--ic_only', dest='ic_only', action='store_true', help='Use all layers for inference')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    args = parser.parse_args()
    return args


def load_test_data():
    global NUM_CHANNELS, MEAN, STD
    if DATASET == 'finetune_mnist':
        train_transform = transforms.Compose([
            # convert to 3 channels
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
    elif DATASET == 'mnist':
        NUM_CHANNELS = 1
        train_transform = transforms.Compose([
            # convert to 3 channels
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    if DATASET == 'finetune_mnist' or DATASET == 'mnist':
        train_data = dset.MNIST(data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.MNIST(data_path,
                                train=False,
                                transform=test_transform,
                                download=True)
    elif DATASET == 'cifar10':
        train_data = dset.CIFAR10(data_path,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10(data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
        
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=1,
                                            pin_memory=True)
    return train_loader, test_loader

def load_model(num_classes, num_channels, ckpt=True):
    net = models.__dict__[ARCH](num_classes, num_channels)
    if ckpt:
        checkpoint = torch.load(chk_path)
        state_tmp = net.state_dict()
        if 'state_dict' in checkpoint.keys():
            state_tmp.update(checkpoint['state_dict'])
        else:
            state_tmp.update(checkpoint)

        #net.load_state_dict(state_tmp)
        model_dict = net.state_dict()
        pretrained_dict = {k:v for k, v in checkpoint['state_dict'].items() if k in model_dict}
        # print(pretrained_dict)
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    return net

def accuracy(output, target, topk=(1, )):
    """Computes the accuracy@k for the specified values of k"""
    # print("In ACCURACY FUNCTION")
    # print("Output shape: ", output.shape)
    # print("Output: ", output)
    # print("Target: ", target)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0)
            correct_k = correct[:k].reshape(-1).float().sum(0)
            
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def denorm(batch):
    # Tensor-ize mean and std
    if not isinstance(MEAN, torch.Tensor):
        mean = torch.tensor(MEAN).view(1, -1, 1, 1)
    if not isinstance(STD, torch.Tensor):
        std = torch.tensor(STD).view(1, -1, 1, 1)
    if use_cuda:
        mean = mean.to(device)
        std = std.to(device)
    return batch * std + mean

def fgsm_sequence(model, data, target, output_branch, adv_examples, epsilon, ic_only=True):
    correct = 0
    loss = 0
    # Get mode prediction
    prediction_counts = Counter()
    for idx in range(len(output_branch)):
        loss += w[idx] * F.cross_entropy(output_branch[idx], target)
        preds = torch.argmax(output_branch[idx], 1)
        prediction_counts[preds.item()] += 1
    
    # Get the mode prediction
    mode_prediction = max(prediction_counts, key=prediction_counts.get)
    
    # Very important!!
    model.zero_grad()
    loss.backward()

    data_grad = data.grad.data
    data_denorm = denorm(data)

    perturbed_data = data_denorm + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    perturbed_data = transforms.Normalize(MEAN, STD)(perturbed_data)

    perturbed_prediction_counts = Counter()
    perturbed_output = model(perturbed_data)
    for idx in range(len(perturbed_output)):
        prec1, prec5 = accuracy(perturbed_output[idx].data, target, topk=(1, 5))
        pred = torch.argmax(perturbed_output[idx], 1)
        perturbed_prediction_counts[pred] += 1

    # Most common prediction
    mode_fin_prediction = max(perturbed_prediction_counts, key=perturbed_prediction_counts.get) if ic_only else pred

    if mode_fin_prediction == target.item():
        correct += 1
        # Special case for saving 0 epsilon examples
        if epsilon == 0 and len(adv_examples) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((mode_prediction, mode_fin_prediction, adv_ex))
    else:
        # Save some adv examples for visualization later
        if len(adv_examples) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((mode_prediction, mode_fin_prediction, adv_ex))
    
    return correct


def validate(val_loader, model, log, epsilon=0.1, ic_only=True):

    print("Validating...")

    # switch to evaluate mode
    model.eval()
    total_correct = 0
    adv_examples = []
    # NOTE: There is no torch.no_grad() here because we need the gradients for FGSM
    # Also note how we do model.zero_grad() before the loss.backward() in the FGSM function
    for i, (input, target) in enumerate(val_loader):
        if i % 50 == 0:
            print_log(f"Validation batch: {i}", log)
            # if i > 0:
            #     break
        if use_cuda:
            target = target.to(device)
            input = input.to(device)
        
        # For adversarial examples
        input.requires_grad = True

        # compute output
        output_branch = model(input)
        # measure accuracy and record loss
        prediction_counts = Counter()   
        for idx in range(len(output_branch)):
            # print(f"Branch {idx} Prec@1: {prec1.item()} Prec@5: {prec5.item()}")
            # print("Output branch w/o data: ", output_branch[idx])
            preds = torch.argmax(output_branch[idx].data, 1)
            prediction_counts[preds.item()] += 1
        prec1, prec5 = accuracy(output_branch[-1].data, target, topk=(1, 5))
   
        # Get the mode prediction or last layer prediction if not ic_only
        mode_prediction = max(prediction_counts, key=prediction_counts.get) if ic_only else preds
        
        # If correct, then do FGSM
        if mode_prediction == target.item():
            total_correct += fgsm_sequence(model, input, target, output_branch, adv_examples, epsilon, ic_only) if epsilon > 0 else 1
        # if i > 10:
        #     break

    final_acc = total_correct/float(len(val_loader))
    print_log(f"Epsilon: {epsilon}\tTest Accuracy = {total_correct} / {len(val_loader)} = {final_acc}", log) 
        
    return final_acc, adv_examples

def plot_adv_examples(epsilons, examples):
    cnt = 0
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title(f"{orig} -> {adv}")
            ex = np.transpose(ex, (1, 2, 0))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    # Save the plot
    print(f"Saving plot to {save_path}/adv_examples.png")
    plt.savefig(f"{save_path}/adv_examples.png")

def plot_accs(epsilons, accuracies):
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .25, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.savefig(f"{save_path}/accs.png")

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def _log_consts(log):
    print_log(f"DATASET: {DATASET}\n", log)
    print_log(f"ARCH: {ARCH}\n", log)
    print_log(f"SEED: {SEED}\n", log)
    print_log(f"BATCH_SIZE: {BATCH_SIZE}\n", log)
    print_log(f"use_cuda: {use_cuda}\n", log)
    print_log(f"device: {device}\n", log)
    print_log(f"NUM_CHANNELS: {NUM_CHANNELS}\n", log)
    print_log(f"NUM_CLASSES: {NUM_CLASSES}\n", log)
    print_log(f"MEAN: {MEAN}\n", log)
    print_log(f"STD: {STD}\n", log)
    print_log(f"chk_path: {chk_path}\n", log)
    print_log(f"weight: {weight}\n", log)
    print_log(f"save_path: {save_path}\n", log)
    print_log(f"ic_only: {ic_only}\n", log)

def main():
    eps = [0.0, 0.05, 0.10, 0.15, 0.20]
    accuracies = []
    examples = []
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    chk_path_first_dir = args.chk_path.split('/')[1]
    args.save_path = f"{args.save_path}/adv_{chk_path_first_dir}"
    # e.g. save_path = ./save_adversarial/save_woROB

    if use_cuda and torch.cuda.is_available():

        torch.cuda.manual_seed_all(args.seed)

    global DATASET, ARCH, SEED, chk_path, save_path, data_path, ic_only
    DATASET = args.dataset
    ARCH = args.arch
    SEED = args.seed
    chk_path = args.chk_path
    save_path = args.save_path
    data_path = args.data_path
    ic_only = args.ic_only

    os.makedirs(args.save_path, exist_ok=True)

    log = open(f"{args.save_path}/adv_ex.txt", "w")
    # ------------------------- SETUP -------------------------
    
    _log_consts(log)

    train_loader, test_loader = load_test_data()
    print(NUM_CHANNELS)
    print(MEAN, STD)

    model = load_model(NUM_CLASSES, NUM_CHANNELS)
    input = next(iter(train_loader))[0]

    print("Input shape: ", input.shape)
    if use_cuda:
        model = model.to(device)
        input = input.to(device)

    for ep in eps:
        print_log(f"Running for epsilon: {ep}", log)
        acc, adv_examples = validate(test_loader, model, log, ep, ic_only)
        accuracies.append(acc)
        examples.append(adv_examples)

    print_log(f"Accuracies: {accuracies}\n", log)
    plot_accs(eps, accuracies)
    plot_adv_examples(eps, examples)

if __name__ == '__main__':
    main()