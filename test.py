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



def subplot(img1, img2, img3):
    fig2 = plt.figure(figsize=(12,12))
    img1 = img1 / 2 + 0.5    
    npimg1 = img1.numpy()
    plt.subplot(311)
    plt.imshow(np.transpose(npimg1, (1, 2, 0)))

    img2 = img2 / 2 + 0.5    
    npimg2 = img2.numpy()
    plt.subplot(312)
    plt.imshow(np.transpose(npimg2, (1, 2, 0)))

    img3 = img3 / 2 + 0.5    
    npimg3 = img3.numpy()
    plt.subplot(313)
    plt.imshow(np.transpose(npimg3, (1, 2, 0)))

    plt.show()

def normalize(img):
    print('normalize:', img.size() )
    _min = torch.min(img)
    _max = torch.max(img)
    return (img - _min) / (_max - _min)

def imshow(img):
    img = img / 2 + 0.5    
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def print_max_min(img):
    _min = torch.min(img)
    _max = torch.max(img)
    print('min:', _min.item(), ', max:', _max.item())

def test(saliency_model_file, num_workers, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='data/', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='data/', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    net = saliency_model()
    net = net.cuda()

    net = torch.load(saliency_model_file)
    padding = 20
    for i, data in enumerate(testloader, 0):

        inputs, labels = data

        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        print_max_min(inputs)

        masks,_ = net(inputs,labels)
        #Original Image
        di_images = torchvision.utils.make_grid(inputs.cpu().data, padding=padding)
        # imshow(torchvision.utils.make_grid(inputs.cpu().data))
        #Mask
        normalize(masks)
        # masks = normalize(masks)
        mask = torchvision.utils.make_grid(masks.cpu().data, padding=padding)
        # imshow(torchvision.utils.make_grid(masks.cpu().data))
        #Image Segmented
        # imshow(torchvision.utils.make_grid((inputs * masks).cpu().data))
        segmented = torchvision.utils.make_grid((inputs*masks).cpu().data, padding=padding)
        subplot(di_images, mask, segmented)
    
def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliencyModelFile", type=str)
    parser.add_argument("--numWorkers", type=int, default=1)
    parser.add_argument("--batchSize", type=int, default=1)

    args = parser.parse_args()
    saliency_model_file = args.saliencyModelFile
    num_workers = args.numWorkers
    batch_size = args.batchSize
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test(saliency_model_file, num_workers, batch_size)
    
__main__()