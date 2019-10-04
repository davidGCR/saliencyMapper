import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable

from scipy import misc


'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class PixelShuffleBlock(nn.Module):
    def forward(self, x):
        return F.pixel_shuffle(x, 2)
      

def SimpleCNNBlock(in_channels, out_channels,
                 kernel_size=3, layers=1, stride=1,
                 follow_with_bn=True, activation_fn=lambda: nn.ReLU(True), affine=True):

        assert layers > 0 and kernel_size%2 and stride>0
        current_channels = in_channels
        _modules = []
        for layer in range(layers):
            _modules.append(nn.Conv2d(current_channels, out_channels, kernel_size, stride=stride if layer==0 else 1, padding=int(kernel_size/2), bias=not follow_with_bn))
            current_channels = out_channels
            if follow_with_bn:
                _modules.append(nn.BatchNorm2d(current_channels, affine=affine))
            if activation_fn is not None:
                _modules.append(activation_fn())
        return nn.Sequential(*_modules)

def SimpleUpsamplerSubpixel(in_channels, out_channels, kernel_size=3, activation_fn=lambda: torch.nn.ReLU(inplace=False), follow_with_bn=True):
    _modules = [
        SimpleCNNBlock(in_channels, out_channels * 4, kernel_size=kernel_size, follow_with_bn=follow_with_bn),
        PixelShuffleBlock(),
        activation_fn(),
    ]
    return nn.Sequential(*_modules)

class UpSampleBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels,passthrough_channels, stride=1):
        super(UpSampleBlock, self).__init__()
        self.upsampler = SimpleUpsamplerSubpixel(in_channels=in_channels,out_channels=out_channels)
        self.follow_up = BasicBlock(out_channels+passthrough_channels,out_channels)

    def forward(self, x, passthrough):
        out = self.upsampler(x)
        out = torch.cat((out,passthrough), 1)
        return self.follow_up(out)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        up_block = UpSampleBlock;
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.uplayer4 = UpSampleBlock(in_channels=512,out_channels=256,passthrough_channels=256)
        self.uplayer3 = UpSampleBlock(in_channels=256,out_channels=128,passthrough_channels=128)
        self.uplayer2 = UpSampleBlock(in_channels=128,out_channels=64,passthrough_channels=64)
        
        self.embedding = nn.Embedding(num_classes,512)
        
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.saliency_chans = nn.Conv2d(64,2,kernel_size=1,bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
            break;
        return nn.Sequential(*layers)

    
    def forward(self, x,labels):
        out = F.relu(self.bn1(self.conv1(x)))
        
        scale1 = self.layer1(out)
        scale2 = self.layer2(scale1)
        scale3 = self.layer3(scale2)
        scale4 = self.layer4(scale3)

      
        em = torch.squeeze(self.embedding(labels.view(-1, 1)), 1)
        act = torch.sum(scale4*em.view(-1, 512, 1, 1), 1, keepdim=True)
        th = torch.sigmoid(act)
        scale4 = scale4*th
        
        
        upsample3 = self.uplayer4(scale4,scale3)
        upsample2 = self.uplayer3(upsample3,scale2)
        upsample1 = self.uplayer2(upsample2,scale1)
        
        saliency_chans = self.saliency_chans(upsample1)
        
        
        out = F.avg_pool2d(scale4, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        a = torch.abs(saliency_chans[:,0,:,:])
        b = torch.abs(saliency_chans[:,1,:,:])
        
        return torch.unsqueeze(a/(a+b), dim=1), out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())






def save_checkpoint(state, filename='sal.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(net,filename='small.pth.tar'):
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['state_dict'])
    
    return net

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import argparse

def imshow(img):
    img = img / 2 + 0.5    
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




def visualize(saliency_model):
    transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    net = ResNet18()
    net = net.cuda()

    net = torch.load(saliency_model)

    for i, data in enumerate(testloader, 0):

        inputs, labels = data

        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        masks,_ = net(inputs,labels)
        #Original Image
        imshow(torchvision.utils.make_grid(inputs.cpu().data))
        #Mask
        imshow(torchvision.utils.make_grid(masks.cpu().data))
        #Image Segmented
        imshow(torchvision.utils.make_grid((inputs*masks).cpu().data))

# def __main__():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--saliencyModel", type=str, default='data/checkpoints/saliency_model_2222.tar')
#     args = parser.parse_args()
#     saliencyModel = args.saliencyModel
#     visualize(saliency_model=saliencyModel)
    
# __main__()