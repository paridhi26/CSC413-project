import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import models
import utils
import math
import copy
import random
from PIL import Image
from torch.utils.data import Dataset
from utils import AverageMeter, RecorderMeter
from tqdm import tqdm
from models.quantization import quan_Conv2d, quan_Linear, quantize

import argparse
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Model
parser.add_argument('--model', default='resnet32', type=str, help='model type')
# Dataset
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
# Paths
parser.add_argument('--data_path', default='./data', type=str, help='data path')
parser.add_argument('--save_path', default='./save', type=str, help='experiment path')
parser.add_argument('--chk_path', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# Run settings
parser.add_argument('--manualSeed', default=0, type=int, help='manual seed')
# Device options
parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')

args = parser.parse_args()

log = open(
    os.path.join(f'{args.save_path}', 'log_seed_{}.txt'.format(args.manualSeed)), 'w')

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

# Print args
print_log(args, log)

if args.dataset == 'mnist':
    num_classes = 10
    num_channels = 1
    mean = (0.5,)
    std = (0.5,)
if args.dataset == 'finetune_mnist':
    num_classes = 10
    num_channels = 3
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

net = models.__dict__[f'{args.model}1'](num_classes, num_channels)
net1 = models.__dict__[args.model](num_classes, num_channels)
pretrain_dict = torch.load(f'../../cifar10/resnet32/{args.chk_path}')
pretrain_dict = pretrain_dict['state_dict']
model_dict = net.state_dict()
pretrained_dict = {str(k): v for k, v in pretrain_dict.items() if str(k) in model_dict}
model_dict.update(pretrained_dict) 

net.load_state_dict(model_dict) 
net.eval()
net=net.cuda()

net1.load_state_dict(model_dict) 
net1.eval()
net1=net1.cuda()


print_log('==> Preparing data..', log)
print_log('==> Preparing data..', log)
if args.dataset == 'finetune_mnist':
        # Convert mnist to 3 channels
        train_transform = transforms.Compose([
            # convert to 3 channels
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean, std)
        ])
else:
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


train_data = torchvision.datasets.MNIST(f'../../cifar10/resnet32/{args.data_path}', train=True, transform=train_transform, download=True)
test_data = torchvision.datasets.MNIST(f'../../cifar10/resnet32/{args.data_path}',
                        train=False,
                        transform=test_transform,
                        download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) 

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)  

for m in net.modules():
    if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
        m.__reset_stepsize__()
        m.__reset_weight__()
        
for m in net1.modules():
    if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
        m.__reset_stepsize__()
        m.__reset_weight__()

        
start = 21
end = 31
I_t = np.load(f'{args.save_path}/SNI.npy')
I_t=torch.Tensor(I_t).long().cuda()
perturbed = torch.load(f'{args.save_path}/perturbed.pth')
print("I_t:", I_t)

n_b = 0
n_e = []
ASR = 0
s_b = []
ASR_t = 90
n_b_max = 500


def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def find_psens(model, data_loader, perturbed):
    model.eval()
    for batch_idx, (data, target) in enumerate(data_loader):
        data,target = data.cuda(), target.cuda()
        data[:,:,start:end,start:end] = perturbed
        y = model(data,nolast = True)[15]
        y[:,I_t] = 10
        break
    ys_target = torch.zeros_like(target)
    ys_target[:] = 2
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    output_nolast = model(data,nolast=True)[15]
    output1 = model(data)[15]
    loss_mse = criterion1(output_nolast,y.detach())
    loss_ce = criterion2(output1,ys_target)
    loss = loss_mse + loss_ce
    model.zero_grad()
    loss.backward()
    F = []
    n = 0
    for m in net1.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            n += 1 
            if m.weight.grad is not None:
                p_grad = m.weight.grad.data.flatten()
                p_weight = m.weight.data.flatten()

                Q_p = torch.max(p_weight)

                # Calculate step for each element in p_grad
                steps = torch.where(p_grad < 0, Q_p - p_weight, 0)

                # Calculate fit for each element in p_grad
                fit = torch.abs(p_grad) * steps

                # Find the maximum fit
                max_fit = torch.max(fit)

                F.append(max_fit)
            else:
                F.append(0)
    idx = F.index(max(F))
    
    return (idx+1)

def identify_vuln_elem(model, psens, data_loader,perturbed, num):
    model.eval()
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx == num:
            data,target = data.cuda(), target.cuda()
            data[:,:,start:end,start:end] = perturbed
            y = model(data,nolast = True)[15]
            y[:,I_t] = 10
            break
    ys_target = torch.zeros_like(target)
    ys_target[:] = 2
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    output_nolast = model(data,nolast=True)[15]
    output1 = model(data)[15]
    loss_mse = criterion1(output_nolast,y.detach())
    loss_ce = criterion2(output1,ys_target)
    loss = loss_mse + loss_ce
    model.zero_grad()
    loss.backward()
    n = 0
    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            n+=1
            if n == psens:
                fit = []
                p_grad = m.weight.grad.data.flatten()
                p_weight = m.weight.data.flatten()
                Q_p = max(p_weight)
                for i in range(len(p_grad)):
                    if p_grad[i] < 0:
                        step = Q_p - p_weight[i]
                    else:
                        step = 0
                    f = abs(p_grad[i])*step
                    fit.append(f)
                break
    index = fit.index(max(fit))
    
    return index

def find_optim_value(model, psens, ele_loc, data_loader, choice, perturbed,num):
    model.eval()
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    n=0
    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            n+=1
            if n == psens:
                p_weight = m.weight.data.flatten()
                p_weight[ele_loc] = choice
                m.weight.data = p_weight.reshape(m.weight.data.shape)
                break
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx == num:
            data,target = data.cuda(), target.cuda()
            pre = model(data)[15]
            loss_ce = criterion2(pre, target)
            data[:,:,start:end,start:end] = perturbed
            y = model(data,nolast = True)[15]
            y[:,I_t] = 10
            break
    ys_target = torch.zeros_like(target)
    ys_target[:] = 2
    output_nolast = model(data,nolast=True)[15]
    output1 = model(data)[15]
    loss_cbs = criterion1(output_nolast,y.detach()) + criterion2(output1,ys_target)
    loss = loss_cbs + 2*loss_ce
    
    return loss

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
    
index_list = []
def validate2(val_loader, model, criterion, num_branch):
    global index_list
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

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
            target = target.cuda()
            input = input.cuda()

            # compute output
            #w = list(map(float, args.weight.split(',')))
            output_branch = model(input)
            #loss = criterion(output, target)
            loss = 0
            for idx in range(len(output_branch)):
                loss += 1 * criterion(output_branch[idx], target)

            # measure accuracy and record loss
            #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            
            for idx in range(len(output_branch)):
                prec1, prec5 = accuracy(output_branch[idx].data, target, topk=(1, 5))
                top1_list[idx].update(prec1, input.size(0)) 
                top5_list[idx].update(prec5, input.size(0))

            
            losses.update(loss.item(), input.size(0))
            # top1.update(prec1.item(), input.size(0))
            # top5.update(prec5.item(), input.size(0))

    c_=0
    
    max_ = 0
    for item in top1_list:
        if item.avg > max_:
            max_ = item.avg 
            index_list.append(c_)
        #print("c_{}", c_, item.avg)  
        c_ += 1 
    return index_list

def validate(val_loader, model, criterion, num_branch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top_list=[]
    for i in range(num_branch):
        top_list.append(AverageMeter())

    exit_b1 = AverageMeter()
    exit_b2 = AverageMeter()
    exit_b3 = AverageMeter()
    exit_b4 = AverageMeter()
    exit_b5 = AverageMeter()
    exit_b6 = AverageMeter()
    exit_m = AverageMeter()

    

    decision = []

    top1_list = []
    for idx in range(num_branch):# acc list for all branches
        top1_list.append(AverageMeter())
    top5_list = []
    for idx in range(num_branch):
        top5_list.append(AverageMeter())
    count_list = [0] * num_branch



    # switch to evaluate mode
    model.eval()
    output_summary = [] # init a list for output summary

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()
            target_var = Variable(target)
        


            
            out_list = [] # out pro
            output_branch = model(input)
            sm = torch.nn.functional.softmax
            for output in output_branch:
                prob_branch = sm(output, dim=1)
                max_pro, indices = torch.max(prob_branch, dim=1)
                out_list.append((prob_branch, max_pro))
            
            num_c = 3#6 # the number of branches 
            branch_index = list(range(0, num_branch))#num_branch
            for j in range(input.size(0)):
                #tar = torch.from_numpy(np.array(target[j]).reshape((-1,1))).squeeze().long().cuda(async=True)
                tar = torch.from_numpy(target[j].cpu().numpy().reshape((-1,1))).squeeze(0).long().cuda()
                tar_var = Variable(torch.from_numpy(target_var.data.cpu().numpy()[j].flatten()).long().cuda())
                pre_index = random.sample(branch_index, num_c) # randomly selected index
                # pre_index = random.sample(index_list, num_c)
                c_ = 0
                for item in sorted(pre_index):#to do: no top 5
                    if out_list[item][1][j] > 0.95 or (c_ + 1 == num_c):
                        #item = -1
                        sm_out = out_list[item][0][j]
                        out = Variable(torch.from_numpy(sm_out.data.cpu().numpy().reshape((1,-1))).float().cuda())
                        loss = criterion(out, tar_var)
                        prec1, = accuracy(out.data, tar, topk=(1,))
                        top1.update(prec1, 1)
                        losses.update(loss.item(), 1)
                        count_list[item]+=1
                        break
                    c_ += 1
        print_log(f"top1.avg:{top1.avg}", log)
        #print("top1.avg:", top1.avg, top5.avg, top_list[0].avg, top_list[1].avg, top_list[2].avg, top_list[3].avg, top_list[4].avg, top_list[5].avg, top_list[6].avg)
        #print(count_list)
        return top1.avg
    
def validate_for_attack(val_loader, model, criterion, num_branch, xh):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top_list=[]
    for i in range(num_branch):
        top_list.append(AverageMeter())

    exit_b1 = AverageMeter()
    exit_b2 = AverageMeter()
    exit_b3 = AverageMeter()
    exit_b4 = AverageMeter()
    exit_b5 = AverageMeter()
    exit_b6 = AverageMeter()
    exit_m = AverageMeter()

    

    decision = []

    top1_list = []
    for idx in range(num_branch):# acc list for all branches
        top1_list.append(AverageMeter())
    top5_list = []
    for idx in range(num_branch):
        top5_list.append(AverageMeter())
    count_list = [0] * num_branch



    # switch to evaluate mode
    model.eval()
    output_summary = [] # init a list for output summary
    # ASR is the rate at which the model is fooled by the trigger
    # See below, target = 2, that's the target label for the trigger
    # The trigger is a square of pixels in the image that somehow makes the model predict the target label

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target[:] =2
            input[:,0:3,start:end,start:end]=xh
            target = target.cuda()
            input = input.cuda()
            target_var = Variable(target, volatile=True)
        


            
            out_list = [] # out pro
            output_branch = model(input)
            sm = torch.nn.functional.softmax
            for output in output_branch:
                prob_branch = sm(output, dim=1)
                max_pro, indices = torch.max(prob_branch, dim=1)
                out_list.append((prob_branch, max_pro))
            
            num_c = 3#6 # the number of branches 
            branch_index = list(range(0, num_branch))#num_branch
            for j in range(input.size(0)):
                #tar = torch.from_numpy(np.array(target[j]).reshape((-1,1))).squeeze().long().cuda(async=True)
                tar = torch.from_numpy(target[j].cpu().numpy().reshape((-1,1))).squeeze(0).long().cuda()
                tar_var = Variable(torch.from_numpy(target_var.data.cpu().numpy()[j].flatten()).long().cuda())
                pre_index = random.sample(branch_index, num_c) # randomly selected index
                # pre_index = random.sample(index_list, num_c)
                c_ = 0
                for item in sorted(pre_index):#to do: no top 5
                    if out_list[item][1][j] > 0.95 or (c_ + 1 == num_c):
                        #item = -1
                        sm_out = out_list[item][0][j]
                        out = Variable(torch.from_numpy(sm_out.data.cpu().numpy().reshape((1,-1))).float().cuda())
                        loss = criterion(out, tar_var)
                        prec1, = accuracy(out.data, tar, topk=(1,))
                        top1.update(prec1, 1)
                        losses.update(loss.item(), 1)
                        count_list[item]+=1
                        break
                    c_ += 1
        print_log(f"top1.asr!:{top1.avg}", log)
        #print("top1.avg:", top1.avg, top5.avg, top_list[0].avg, top_list[1].avg, top_list[2].avg, top_list[3].avg, top_list[4].avg, top_list[5].avg, top_list[6].avg)
        print_log(f"Count_list:{count_list}", log)
        return top1.avg

from bitstring import Bits
def countingss(param,param1):
    #param = quantize(fpar,step,lvls)
    #param1 = quantize(fpar1,step,lvls)
    count = 0
    b1=Bits(int=int(param), length=8).bin
    b2=Bits(int=int(param1), length=8).bin
    for k in range(8):
        diff=int(b1[k])-int(b2[k])
        if diff!=0:
            count=count+1
    return count

def test1(model, loader, xh):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    targets = 2
    with torch.no_grad():
        for x, y in loader:
            x_var = to_var(x, volatile=True)
            x_var[:,0:3,start:end,start:end]=xh
            y[:]=targets 
        
            scores = model(x_var)[15]
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == y).sum()

    asr = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the trigger data' 
        % (num_correct, num_samples, 100 * asr))

    return asr


criterion = nn.CrossEntropyLoss()
criterion=criterion.cuda()
validate2(test_loader, net1, criterion, 16)
print_log(f"Index_list:{index_list}", log)
validate(test_loader, net1, criterion, 16)
validate_for_attack(test_loader, net1, criterion, 16, perturbed)
psens = find_psens(net1,test_loader,perturbed)
print("Psens", psens)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) 
num=0
last_loc = 0

dpi = 80
width, height = 1200, 800
legend_fontsize = 10
scale_distance = 48.8
figsize = width / float(dpi), height / float(dpi)
fig = plt.figure(figsize=figsize)
x_axis = []
y_axis = []
acc = []

# Attack budget of 50 bit flips

while n_b<50:
    n=0
    for m in net1.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            n+=1
            if n == psens:
                p_weight = m.weight.data.flatten()
                R = max(abs(p_weight))
                break

    k = math.floor(R*2/10)
    #k = R*2/20
    point = []
    for i in range(10):
        point.append(R-k*(i))
        
    ele_loc = identify_vuln_elem(net1,psens,test_loader,perturbed,num)
    if ele_loc == last_loc:
        num+=1
    if num == 8:
        num = 0
    last_loc = ele_loc
    n_e.append(ele_loc)
    old_elem = copy.deepcopy(p_weight[ele_loc])
    loss_troj = []
    for i in range(len(point)):
        loss = find_optim_value(net1, psens, ele_loc, test_loader, point[i], perturbed,num)
        #print(loss)
        loss_troj.append(loss)
    idx = loss_troj.index(min(loss_troj))
    new_elem = point[idx]
    #if new_elem < old_elem:
    #    new_elem = old_elem
    print("Elems:", new_elem,old_elem)
    n_b += countingss(old_elem, new_elem)
    n=0
    for m in net1.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            n+=1
            if n == psens:
                p_weight = m.weight.data.flatten() 
                p_weight[ele_loc] = new_elem
                m.weight.data = p_weight.reshape(m.weight.data.shape)
                break
    ASR = validate_for_attack(test_loader, net1, criterion, 16, perturbed)
    test1(net1,test_loader,perturbed)
    print_log(f"n_b:{n_b}", log)
    x_axis.append(n_b)
    y_axis.append(ASR.cpu())
    plt.xlabel('bit_flips', fontsize=16)
    plt.ylabel('asr', fontsize=16)
    plt.plot(x_axis,y_axis)
    fig.savefig(f'{args.save_path}/asr.png', dpi=dpi, bbox_inches='tight')
    
validate(test_loader, net1, criterion, 16)
print("n_b:", n_b)