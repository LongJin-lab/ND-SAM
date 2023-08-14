import argparse
import torch
import numpy as np
import sys; sys.path.append("..")
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.logger import Logger
from utility.eval import accuracytopk
from utility.misc import AverageMeter
from model import *
import torchvision
import torchvision.transforms as transforms

from original.nd import ND
from original.ndsam import NDSAM

from original.sam import SAM


torch.manual_seed(0)
np.random.seed(0)

logger = Logger('sam_bosielabel_wrn.txt', title='cifar100')
logger.set_names(['Valid Loss', 'Valid Acc.', 'Accuracy_top5'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.1, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument('--dampening', default=0.8, type=float, help='dampening')
    # parser.add_argument('--arch', '-a', default='ResNet18', type=str)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset = Cifar(args.batch_size, args.threads)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='/home/aigroup/lab415/ds/nd_sam/data', train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root='/home/aigroup/lab415/ds/nd_sam/data', train=False, download=False, transform=transform_test)
    
    # noise_ratio = 0.2  # 噪声比例
    # num_samples = len(trainset.targets)
    # num_noisy_samples = int(noise_ratio * num_samples)
    # noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
    # noisy_labels = np.random.randint(0, 10, size=num_noisy_samples)
    # for i, idx in enumerate(noisy_indices):
    #     trainset.targets[idx] = noisy_labels[i]
    # 训练参数
    epsilon = 10.0  # 隐私预算
    sensitivity = 1.0  # 敏感度

    def add_noise(data):
        noise_scale = sensitivity / epsilon
        noise = np.random.laplace(loc=0, scale=noise_scale, size=data.shape)
        noisy_data = data + noise
        return noisy_data
    
    trainloader = torch.utils.data.DataLoader(
        trainset, args.batch_size, shuffle=True, num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(
        testset, args.batch_size, shuffle=False, num_workers=args.workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    log = Log(log_each=10)
    # net_name = args.arch
    # model_name = args.arch
    # if args.arch == "r18":
    #     net = resnet18()
    # elif args.arch == "r20":
    #     net = resnet20()
    # elif args.arch == "r34":
    #     net = resnet34()
    # elif args.arch == "r34":
    #     net = resnet34()
    # elif args.arch == "r50":
    #     net = resnet50()
    # elif args.arch == "r101":
    #     net = resnet101()
    # elif args.arch == "r152":
    #     net = resnet152()
    # elif args.arch == "m":
    #     net = mobilenet()
    # elif args.arch == "mv2":
    #     net = mobilenetv2()
    # elif args.arch == "iv3":
    #     net = inceptionv3()
    # elif args.arch == "pr18":
    #     net = preactresnet18()
    # elif args.arch == "pr34":
    #     net = preactresnet34()
    # elif args.arch == "pr50":
    #     net = preactresnet50()
    # elif args.arch == "pr101":
    #     net = preactresnet101()
    # elif args.arch == "pr152":
    #     net = preactresnet152()
    # elif args.arch == "googlenet":
    #     net = googlenet()
    # elif args.arch == "vgg":
    #     net = vgg19_bn()
    # elif args.arch == "densnet":
    #     net = densenet121()
    # net = net.to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True
    net = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=100).to(device)

    # base_optimizer = ND
    # optimizer = NDSAM(net.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate,momentum=args.momentum,weight_decay=args.weight_decay)
    
    base_optimizer = torch.optim.SGD
    optimizer = SAM(net.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    

    
    for epoch in range(args.epochs):
        net.train()
        log.train(len_dataset=len(trainset))
    
        val_acc_log = AverageMeter()
        
        for batch in trainloader:
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            enable_running_stats(net)
            # add dp noise into input data
            noisy_inputs = add_noise(inputs.cpu().numpy())

            # transfer to Tensor
            noisy_inputs = torch.from_numpy(noisy_inputs).to(device, dtype=torch.float32)
            predictions = net(noisy_inputs.to(device))
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(net)
            smooth_crossentropy(net(noisy_inputs.to(device)), targets, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(net, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        net.eval()
        log.eval(len_dataset=len(testset))

        with torch.no_grad():
            for batch in testloader:
                inputs, targets = (b.to(device) for b in batch)

                predictions = net(inputs)
                loss = smooth_crossentropy(predictions, targets)
                # val_loss_log.update(loss.item(), inputs.size(0))
                correct = torch.argmax(predictions, 1) == targets
                log(net, loss.cpu(), correct.cpu())
                prec1, prec5 = accuracytopk(predictions.data, targets.data, topk=(1, 5))
                val_acc_log.update(prec5.item(), inputs.size(0))

            loss_formatted = round(log.get_loss()/10000, 4)
            logger.append([str(loss_formatted), str(log.get_accuracy()/100), str(val_acc_log.avg)])


    log.flush()
