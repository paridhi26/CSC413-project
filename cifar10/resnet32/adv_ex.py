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

root = './data'
download = True
DATASET = 'finetune_mnist'
data_path = './data'
ARCH = "resnet32_quan"
chk_path = './save_finetune/cifar60/model_best.pth.tar'
BATCH_SIZE = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda = torch.cuda.is_available()
save_path = './save_adversarial'
SEED = 42
NUM_CHANNELS = 3
NUM_CLASSES = 10
MEAN, STD = (0.5,), (0.5,)
weight='1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1'
w = [float(i) for i in weight.split(',')]

random.seed(SEED)
torch.manual_seed(SEED)

if use_cuda:
    torch.cuda.manual_seed_all(SEED)

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

def load_test_data():

    if DATASET == 'finetune_mnist':
        train_transform = transforms.Compose([
            # convert to 3 channels
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(MEAN, STD)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(MEAN, STD)
        ])
    if DATASET == 'finetune_mnist':
        train_data = dset.MNIST(data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.MNIST(data_path,
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

def _get_avg(lst):
    sum = 0
    for item in lst:
        sum += item.avg
    return sum/len(lst)

def denorm(batch):
    # Tensor-ize mean and std
    if not isinstance(MEAN, torch.Tensor):
        MEAN = torch.tensor(MEAN).view(1, 3, 1, 1)
    if not isinstance(STD, torch.Tensor):
        STD = torch.tensor(STD).view(1, 3, 1, 1)
    if use_cuda:
        MEAN = MEAN.cuda()
        STD = STD.cuda()
    return batch * STD + MEAN

def fgsm_sequence(model, data, target, output_branch, adv_examples, epsilon, threshold=0.9):
    correct = 0
    loss = 0
    # Get mode prediction
    prediction_counts = Counter()
    for idx in range(len(output_branch)):
        loss += w[idx] * F.cross_entropy(output_branch[idx], target)
        _, preds = torch.argmax(output_branch[idx], 1)
        prediction_counts[preds.item()] += 1
    
    # Get the mode prediction
    mode_prediction = max(prediction_counts, key=prediction_counts.get)
    
    model.zero_grad()
    loss.backward()

    data_grad = data.grad.data
    data_denorm = denorm(data)

    perturbed_data = data_denorm + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    perturbed_data = (perturbed_data - MEAN) / STD

    top1_list = []
    perturbed_prediction_counts = Counter()
    perturbed_output = model(perturbed_data)
    for idx in range(len(perturbed_output)):
        prec1, prec5 = accuracy(perturbed_output[idx].data, target, topk=(1, 5))
        perturbed_prediction_counts[torch.argmax(perturbed_output[idx], 1).item()] += 1
        top1_list.append(prec1)

    # Most common prediction
    mode_fin_prediction = max(perturbed_prediction_counts, key=perturbed_prediction_counts.get)
    top_1_avg = 0
    for item in top1_list:
        top_1_avg += item
    top_1_avg /= len(top1_list)

    # TODO: Add the mode prediction to get a sense of the change in prediction
    if top_1_avg > threshold:
        correct += 1
        # Special case for saving 0 epsilon examples
        if epsilon == 0 and len(adv_examples) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((mode_prediction.item(), mode_fin_prediction.item(), adv_ex))
    else:
        # Save some adv examples for visualization later
        if len(adv_examples) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((mode_prediction.item(), mode_fin_prediction.item(), adv_ex))
    
    return correct


def validate(val_loader, model, num_branch, ic_only, epsilon=0.1, threshold=0.9):
    top1 = AverageMeter()
    top5 = AverageMeter()

    print("Validating...")

    top1_list = []
    for idx in range(num_branch):
        top1_list.append(AverageMeter())
    top5_list = []
    for idx in range(num_branch):
        top5_list.append(AverageMeter())

    # switch to evaluate mode
    model.eval()
    total_correct = 0
    adv_examples = []

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if use_cuda:
                target = target.cuda(non_blocking=True)
                input = input.cuda()
            
            # For adversarial examples
            input.requires_grad = True

            # compute output
            output_branch = model(input)
            # measure accuracy and record loss
            #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            
            for idx in range(len(output_branch)):
                prec1, prec5 = accuracy(output_branch[idx].data, target, topk=(1, 5))
                # print(f"Branch {idx} Prec@1: {prec1.item()} Prec@5: {prec5.item()}")
                # print("Output branch w/o data: ", output_branch[idx])
                top1_list[idx].update(prec1, input.size(0)) 
                top5_list[idx].update(prec5, input.size(0))

            if ic_only:
                top1_avg = _get_avg(top1_list[:-1])
                top5_avg = _get_avg(top5_list[:-1])
            else:
                top1_avg = top1_list[-1].avg
                top5_avg = top5_list[-1].avg
            if top1_avg > threshold * 100:
                total_correct += fgsm_sequence(model, input, target, output_branch, adv_examples, epsilon, threshold)

        final_acc = total_correct/float(len(val_loader))
        print(f"Epsilon: {epsilon}\tTest Accuracy = {total_correct} / {len(val_loader)} = {final_acc}")  

        print(
        '  **Test** Prec_B1@1 {top1_b1.avg:.3f} Prec_B1@5 {top5_b1.avg:.3f} Error@1 {error1:.3f}'
        '  **Test** Prec_B2@1 {top1_b2.avg:.3f} Prec_B2@5 {top5_b2.avg:.3f} Error@1 {error2:.3f}'
        '  **Test** Prec_B3@1 {top1_b3.avg:.3f} Prec_B3@5 {top5_b3.avg:.3f} Error@1 {error3:.3f}'
        '  **Test** Prec_B4@1 {top1_b4.avg:.3f} Prec_B4@5 {top5_b4.avg:.3f} Error@1 {error4:.3f}'
        '  **Test** Prec_B5@1 {top1_b5.avg:.3f} Prec_B5@5 {top5_b5.avg:.3f} Error@1 {error5:.3f}'
        '  **Test** Prec_B6@1 {top1_b6.avg:.3f} Prec_B6@5 {top5_b6.avg:.3f} Error@1 {error6:.3f}'
        '  **Test** Prec_Bmain@1 {top1_main.avg:.3f} Prec_Bmain@5 {top5_main.avg:.3f} Error@1 {errormain:.3f}'
        .format(top1_b1=top1_list[0], top5_b1=top5_list[0], error1=100 - top1_list[0].avg,
                top1_b2=top1_list[1], top5_b2=top5_list[1], error2=100 - top1_list[1].avg,
                top1_b3=top1_list[2], top5_b3=top5_list[2], error3=100 - top1_list[2].avg,
                top1_b4=top1_list[3], top5_b4=top5_list[3], error4=100 - top1_list[3].avg,
                top1_b5=top1_list[4], top5_b5=top5_list[4], error5=100 - top1_list[4].avg,
                top1_b6=top1_list[5], top5_b6=top5_list[5], error6=100 - top1_list[5].avg,
                top1_main=top1_list[-1], top5_main=top5_list[-1], errormain=100 - top1_list[-1].avg,
        ))
        
    sum=0
    if ic_only: #if only train ic branch
        for item in top1_list[:-1]:
            sum += item.avg
        top1_avg = sum/len(top1_list[:-1])
        sum=0
        for item in top5_list[:-1]:
            sum += item.avg
        top5_avg = sum/len(top5_list[:-1])
    else:
        top1_avg = top1_list[-1].avg
        top5_avg = top5_list[-1].avg

    return top1_avg, top5_avg, final_acc, adv_examples

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
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{save_path}/mnist_adv_examples.png")
    plt.show()

def main():
    eps = [0.0, 0.1]
    accuracies = []
    examples = []
    train_loader, test_loader = load_test_data()

    model = load_model(NUM_CLASSES, NUM_CHANNELS, False)
    input = next(iter(test_loader))[0]

    print("Input shape: ", input.shape)
    if use_cuda:
        model = model.cuda()
        input = input.cuda()

    branch_out = model(input)
    length = len(branch_out)

    for ep in eps:
        top1_avg, top5_avg, acc, adv_examples = validate(test_loader, model, length, True, ep)
        accuracies.append(acc)
        examples.append(adv_examples)

    print("Top1: ", top1_avg)
    print("Top5: ", top5_avg)

    print("Accuracies: ", accuracies)
    plot_adv_examples(eps, examples)

if __name__ == '__main__':
    main()