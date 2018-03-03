#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.nn.parallel

import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import os
import sys
import math

import shutil

import setproctitle

import resnext
#import make_graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=128)
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
    parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or '/home/ywang/workspace/senet_multiscale/senet.base'
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
#    os.makedirs(args.save, exist_ok=True)
    os.makedirs(args.save)

    best_epoch=100.0

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    traindir = os.path.join('/home/ywang/data', 'ILSVRC2012_img_train')
    valdir = os.path.join('/home/ywang/data', 'ILSVRC2012_img_val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = DataLoader(
        dset.ImageFolder(traindir, transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(
		brightness = 0.4,
                contrast = 0.4,
                saturation = 0.4,
       	    ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batchSz, shuffle=True, **kwargs)
 
    val_loader = DataLoader(
        dset.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batchSz, shuffle=False, **kwargs)


    net = resnext.resnext101(baseWidth=args.base_width,cardinality=args.cardinality)
    net = torch.nn.DataParallel(net).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
#    if args.cuda:
#        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=6e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, train_loader, criterion, optimizer, trainF)
        epoch_error = test(args, epoch, net, val_loader, criterion, optimizer, testF)
        torch.save(net, os.path.join(args.save, 'latest.pth'))
        if epoch_error < best_epoch:
            best_epoch = epoch_error
            torch.save(net, os.path.join(args.save, 'best.pth'))
        os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()

def train(args, epoch, net, train_loader, criterion, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(train_loader.dataset)
    itersize = 8
    optimizer.zero_grad()


    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
#        optimizer.zero_grad()
        output = net(data)
        loss1 = criterion(output, target)
        loss = loss1/itersize
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        if (batch_idx+1) % itersize == 0:
            optimizer.step()
            optimizer.zero_grad()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(train_loader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(train_loader),
            loss1.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss1.data[0], err))
        trainF.flush()

def test(args, epoch, net, val_loader, criterion, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in val_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(val_loader) # loss function already averages over batch size
    nTotal = len(val_loader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.2f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()
    return err

def adjust_opt(optAlg, optimizer, epoch):
    lr = 0.6 * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__=='__main__':
    main()
