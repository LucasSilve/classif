'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import scipy
import numpy as np
from numpy.fft import fftshift
from scipy.fftpack import fft2
import matplotlib.pyplot as plt
import torch.optim as optim
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.autograd import Variable


#from utils import progress_bar

channel_=3
padding_=1
kernel=2*padding_+1
imsize=32
num_classes=10


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    #cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    #cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, (1024, 2), 1024]
    cfg = [16, (32, 2), 32, 32, (64,2), 64,64,64,64,64,64,64,64]
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(channel_,8 , kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.layers = self._make_layers(in_planes=8)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNet()
    x = torch.randn(1,3,8,8)
    y = net(x)
    print('testing :')
    print(y.size())

test()


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





"""class Block(nn.Module):
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


        return out"""



#net=Net(Block,[2,2,2,5,10])


batch=1

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                         shuffle=False, num_workers=2)


#initialisation

lum_seul=True
if lum_seul:
    channel=1
else: channel=3
nombre_filtre=16
padding_=5
kernel=2*padding_+1
learning_rate=0.05
lambda_regular=1


def pad(f):                                         #effectue un padding pour le filtre pour pouvoir faire la transformee de Fourier
    out=np.zeros((32,32), dtype=complex)
    out[15-padding_:16+padding_,15-padding_:16+padding_]=f
    return out

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0, fft_shift=None):
    gab = np.zeros((M, N), np.complex64)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)

    for ex in [0]:
        for ey in [0]:
            [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
            arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab = gab + np.exp(arg)

    norm_factor = (2 * 3.1415 * sigma * sigma / slant)
    gab = gab / norm_factor

    if (fft_shift):
        gab = np.fft.fftshift(gab, axes=(0, 1))
    return gab

phi=gabor_2d(kernel,kernel,3,0,0,fft_shift=False,offset=-padding_)
phi=np.absolute(phi)
phi_chap=pad(phi)
phi_chap=fft2(phi_chap)
phi_chap=fftshift(phi_chap)
phi_chap=np.absolute(phi_chap)

fig, (ax1, ax2) = plt.subplots(ncols=2)

img1 = ax1.imshow(phi)
colorbar(img1)

img2 = ax2.imshow(phi_chap)
colorbar(img2)
plt.tight_layout(h_pad=1)
plt.show()


print('ok')
phi=torch.from_numpy(phi)
print(torch.sum(phi))
phi=phi/torch.sum(phi)


avg=nn.Conv2d(nombre_filtre,nombre_filtre,kernel,bias=False,padding=padding_,groups=nombre_filtre)
avg=avg.cuda()

for k1 in range(0,nombre_filtre):
    avg.weight.data[k1,0,:,:]=phi


class Net1(nn.Module):
    def __init__(self, nombre_filtre):
        super(Net1, self).__init__()
        self.conv_real = nn.Conv2d(channel, nombre_filtre, kernel, bias=False,padding=padding_)
        self.conv_imag = nn.Conv2d(channel, nombre_filtre, kernel, bias=False, padding=padding_)

        self.avg = torch.nn.AvgPool2d(5,stride=1,padding=2,count_include_pad=True)
        self.bn = nn.BatchNorm2d(nombre_filtre)
    def forward(self, x):
        y_r = self.conv_real(x)
        y_i = self.conv_imag(x)

        y_r = y_r ** 2
        y_i = y_i ** 2
        y = y_r + y_i
        y = torch.sqrt(y)
        y = avg(y)

        return y


Net1 = Net1(nombre_filtre).cuda()
poids=torch.load('/home/lucas/Pycharm/pycharm-2018.1.4/bin/lucas_code_beginning/save/WN.pth.tar')
Net1.load_state_dict(poids)

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
        if lum_seul:
            inputs = torch.sum(inputs, 1, keepdim=True)
            aux = torch.ones(1)
            aux = 3 * aux
            aux = torch.sqrt(aux)
            aux = aux.item()
            inputs = inputs / aux
        compressedinputs = Net1(inputs)
        outputs = net(compressedinputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimi.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        """progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))"""

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

           #""" progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
           #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))"""

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
