import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from resnet import *
from model import *
import argparse

num_workers = 4
batch_size = 4

def imshow(img):
    img = img / 2 + 0.5    
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test(saliency_model_file):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='data/', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='data/', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    net = saliency_model()
    net = net.cuda()

    net = torch.load(saliency_model_file)

    for i, data in enumerate(testloader, 0):

        inputs, labels = data

        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        masks,_ = net(inputs,labels)
        #Original Image
        imshow(torchvision.utils.make_grid(inputs.cpu().data))
        #Mask
        imshow(torchvision.utils.make_grid(masks.cpu().data))
        #Image Segmented
        imshow(torchvision.utils.make_grid((inputs * masks).cpu().data))
    
def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliencyModelFile", type=str, default='data/checkpoints/saliency_model_2222.tar')
    args = parser.parse_args()
    saliency_model_file = args.saliencyModelFile
    test(saliency_model_file)
    
__main__()