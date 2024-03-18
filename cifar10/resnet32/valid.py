import torch
import os, sys
import models
import pandas as pd
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random
import numpy as np
from utils import AverageMeter, RecorderMeter

SEED = 42
ARCH = 'resnet32_quan'
NUM_CLASSES = 10
IC_ONLY = False
data_path='./data'
MEAN, STD = (0.5,), (0.5,)
criterion = torch.nn.CrossEntropyLoss()
CUDA = False
save_path = './evaluation_saves'
weight='1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1'

random.seed(SEED)
torch.manual_seed(SEED)

if CUDA:
    torch.cuda.manual_seed_all(SEED)

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def load_data():
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4), # simply pads the image
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    train_data = dset.MNIST(data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
    test_data = dset.MNIST(data_path,
                            train=False,
                            transform=test_transform,
                            download=True)
    train_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=1,
                                              pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=1,
                                              pin_memory=True)
    
    return train_loader, test_loader

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
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

def validate(val_loader, model, criterion, log, num_branch, ic_only, summary_output=False):
    losses = AverageMeter()
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
    output_summary = [] # init a list for output summary

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if CUDA:
                target = target.cuda(non_blocking=True)
                input = input.cuda()

            # compute output
            w = list(map(float, weight.split(',')))
            # len 23
            output_branch = model(input)
            #loss = criterion(output, target)
            loss = 0
            for idx in range(len(output_branch)):
                loss += w[idx] * criterion(output_branch[idx], target)

            
            # summary the output
            if summary_output:
                output_branch_arr = np.array(output_branch)
                tmp_list = output_branch_arr.max(1, keepdims=True)[1].flatten() # get the index of the max log-probability
                output_summary.append(tmp_list)

            # measure accuracy and record loss
            #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            
            for idx in range(len(output_branch)):
                prec1, prec5 = accuracy(output_branch[idx].data, target, topk=(1, 5))
                top1_list[idx].update(prec1, input.size(0)) 
                top5_list[idx].update(prec5, input.size(0))

            
            losses.update(loss.item(), input.size(0))
            # top1.update(prec1.item(), input.size(0))
            # top5.update(prec5.item(), input.size(0))

       
        
        print_log(
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
        ), log)
        
    if summary_output:
        output_summary = np.asarray(output_summary).flatten()
        return top1.avg, top5.avg, losses.avg, output_summary

    
def main():
    log = open(
        os.path.join(save_path, 'log_seed_{}.txt'.format(SEED)),
        'w')
    
    
    net = models.__dict__[ARCH](NUM_CLASSES)
    checkpoint = torch.load('./save/model_best.pth.tar')

    recorder = checkpoint['recorder']

    state_tmp = net.state_dict()
    if 'state_dict' in checkpoint.keys():
        state_tmp.update(checkpoint['state_dict'])
    else:
        state_tmp.update(checkpoint)
    
    #net.load_state_dict(state_tmp)
    model_dict = net.state_dict()
    pretrained_dict = {k:v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)

    if CUDA:
        net.cuda()
        criterion.cuda()

    net.load_state_dict(model_dict)
    train_loader, test_loader = load_data()

    input, target = next(iter(train_loader))
    if CUDA: 
        input = input.cuda()

    output_branch = net(input)
    num_branch = len(output_branch) # the number of branches

    print("Evaluating:")
    _,_,_, output_summary = validate(test_loader, net, criterion, log, num_branch, IC_ONLY, summary_output=True)

    pd.DataFrame(output_summary).to_csv(
        os.path.join(save_path, 'output_summary_{}.csv'.format(ARCH)),
        header=['top-1 output'], index=False
    )

if __name__ == '__main__':
    main()