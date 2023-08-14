import argparse
import shutil    
import sys; sys.path.append("..")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn 
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast, GradScaler
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import time
import os
from torch.autograd import Variable
import sys
import numpy as np

from model import *
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.logger import Logger
# from utility.eval import accuracytopk
# from utility.misc import AverageMeter
# from utility.initialize import initialize
# from utility.step_lr import StepLR
# from utility.log import Log
# from data_loader import data_loader

from original.nd import ND
# from original.ndsam import NDSAM
# from original.sam import SAM
from differential_privacy.ndsamdp import NDSAMDP
from differential_privacy.samdp import SAMDP

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))                


logger = Logger('sam_imagenet_r18.txt', title='imagenet')
logger.set_names(['Valid Loss', 'Valid Acc.', 'Accuracy_top5'])

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument("--rho", default=0.1, type=int, help="Rho parameter for SAM.")
parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")

args = parser.parse_args()


best_prec1 = 0
# epsilon = 10.0  # 隐私预算
# sensitivity = 1.0  # 敏感度

# def add_noise(data):
#     noise_scale = sensitivity / epsilon
#     noise = np.random.laplace(loc=0, scale=noise_scale, size=data.shape)
#     noisy_data = data + noise
#     return noisy_data

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))      
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
 
    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)           
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        # Wrap model in DistributedDataParallel (CUDA only for the moment)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    # base_optimizer = ND
    # optimizer = NDSAMDP(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    base_optimizer = torch.optim.SGD
    optimizer = SAMDP(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
   
    if args.resume:                                                                                                                                                                                      
        if os.path.isfile(args.resume):                                 
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)                      
            args.start_epoch = checkpoint['epoch']                     
            best_prec1 = checkpoint['best_prec1']                       
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])        
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    #normalize: - mean / std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],       
                                     std=[0.229, 0.224, 0.225])

    # ImageFolder
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),                      
            transforms.RandomHorizontalFlip(),    
            transforms.ToTensor(),                                             
            normalize,
        ]))

#######
    if args.distributed:
        # Use a DistributedSampler to restrict each process to a distinct subset of the dataset.
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
######

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([ 
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)         # default workers = 4

##### 
    if args.evaluate:
        validate(val_loader, model, criterion)         
        return

##### 
    for epoch in range(args.start_epoch, args.epochs):
        # Use .set_epoch() method to reshuffle the dataset partition at every iteration
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)     
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, prec5, val_loss = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        
        # logger.append([str(prec1), str(prec5), str(val_loss)])
        logger.append(['{:.3f}'.format(val_loss), '{:.3f}'.format(prec1), '{:.3f}'.format(prec5)])

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # target = target.cuda(non_blocking=True)
        # input = input.cuda(non_blocking=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target = target.cuda(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # noisy_inputs = add_noise(input_var.cpu().numpy())
        # noisy_inputs = torch.from_numpy(noisy_inputs.astype(np.float32))
        # noisy_inputs = torch.from_numpy(noisy_inputs).cuda()
        
        enable_running_stats(model)
        # compute output
        output = model(input_var)
        # criterion
        loss = criterion(output, target_var)        

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        # compute gradient and do SGD step
        # optimizer.zero_grad()
        loss.backward()
        optimizer.first_step(zero_grad=True)
        
        # second forward-backward step
        disable_running_stats(model)
        criterion(model(input_var), target_var).backward()
        optimizer.second_step(zero_grad=True)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:     # default=10
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        # logger.append([str(losses.avg), str(top1.avg.item()), str(top5.avg.item())])


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)
        with torch.no_grad():
            input_var = input
            target_var = target


        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg,losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    # Computes and stores the average and current value
    def __init__(self):
        self.reset()       # __init__():reset parameters

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))            # args.lr = 0.1 , 即每30步，lr = lr /10
    for param_group in optimizer.param_groups:       # 将更新的lr 送入优化器 optimizer 中，进行下一次优化
        param_group['lr'] = lr

# 计算准确度
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    maxk = max(topk)
    # size函数：总元素的个数
    batch_size = target.size(0)

    # topk函数选取output前k大个数
    _, pred = output.topk(maxk, 1, True, True)
    ##########不了解t()
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
