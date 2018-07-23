'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.02, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()



channel_=3
padding_=1
kernel=2*padding_+1
imsize=32
num_classes=10

class Block(nn.Module):
    def __init__(self,channel):

        super(Block,self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel,padding=padding_,bias=False)
        #self.pool = F.max_pool2d(3,kernel,stride=1,padding=1)
        self.bn=nn.BatchNorm2d(channel)

    def forward(self,inputs):
        out=F.relu(self.bn(self.conv(inputs)))

        return out


class Net(nn.Module):
    def __init__(self,block,num_blocks,num_classes=10):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel,padding=padding_,bias=False)
        self.conv2 = nn.Conv2d(4, 8, kernel,padding=padding_,bias=False)
        self.conv3 = nn.Conv2d(8, 16, kernel,padding=padding_,bias=False)
        self.conv4 = nn.Conv2d(16,32,kernel,padding=padding_,bias=False)
        self.conv5 = nn.Conv2d(32,64,kernel,padding=padding_,bias=False)


        self.bn1=nn.BatchNorm2d(4)
        self.bn2=nn.BatchNorm2d(8)
        self.bn3=nn.BatchNorm2d(16)
        self.bn4=nn.BatchNorm2d(32)
        self.bn5=nn.BatchNorm2d(64)

        self.linear=nn.Linear(64,num_classes)
        self.maxpool=nn.MaxPool2d(2, stride=2)
        self.block=block
        self.num_blocks=num_blocks
        self.num_classes=num_classes
        self.layer1=self._make_layer(4,0)
        self.layer2=self._make_layer(8,1)
        self.layer3=self._make_layer(16,2)
        self.layer4=self._make_layer(32,3)
        self.layer5=self._make_layer(64,4)


    def _make_layer(self,channel,step):
        layers = []
        for numblocks in range(0,self.num_blocks[step]):
            layers.append(self.block(channel))

        return nn.Sequential(*layers)

    def forward(self,t):
        out=F.relu(self.bn1(self.conv1(t)))
        out=self.layer1(out)
        out=self.maxpool(out)

        out = F.relu(self.bn2(self.conv2(out)))
        out = self.layer2(out)
        out=self.maxpool(out)


        out = F.relu(self.bn3(self.conv3(out)))
        out = self.layer3(out)
        out=self.maxpool(out)

        out = F.relu(self.bn4(self.conv4(out)))
        out = self.layer4(out)
        out = self.maxpool(out)

        out = F.relu(self.bn5(self.conv5(out)))
        out = self.layer5(out)
        out = self.maxpool(out)


        out = out.view(out.size(0), -1)


        out=self.linear(out)


        return out



#net=Net(Block,[2,2,2,5,10])



net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
print(args.lr)
# Training
def train(epoch):

    optimi=optimizer
    if epoch>=50:
        optimi= optim.SGD(net.parameters(), lr=args.lr/(1+epoch), momentum=0.9, weight_decay=5e-4)
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimi.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimi.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):

    train(epoch)
    test(epoch)
