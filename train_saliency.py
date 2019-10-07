import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from scipy import misc
from model import saliency_model
from resnet import *
from loss import Loss
import argparse
import os



def save_checkpoint(state, filename='sal.pth.tar'):
    print('saving ', filename, '...')
    torch.save(state, filename)

def load_checkpoint(net,optimizer,filename='small.pth.tar'):
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return net,optimizer

def cifar10(batch_size, num_workers):
    
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

    return trainloader,testloader,classes

from tqdm import tqdm


def train(batch_size, num_workers, regularizers, device, checkpoint_file):
    num_epochs = 3
    trainloader,testloader,classes = cifar10(batch_size, num_workers)

    net = saliency_model()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    # black_box_func = resnet(pretrained=True)
    # black_box_func = black_box_func.cuda()
    model_name = 'alexnet'
    # defaults.device = 'cpu'
    black_box_func = BlackBoxModel(model_name=model_name, pretrained=True, num_classes=10)
    black_box_func.load('data/checkpoints/black_box_model_alexnet.tar',device='cpu')
    black_box_func.toDevice(device)
    black_box_func = black_box_func.getModel()

    
    loss_func = Loss(num_classes=10,regularizers= regularizers)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("----- Epoch {}/{}".format(epoch, num_epochs))
        running_loss = 0.0
        running_corrects = 0.0
        running_loss_train = 0.0
        
        for i, data in tqdm(enumerate(trainloader, 0)):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()

            mask,out = net(inputs,labels)
            # print('-----mask shape:',mask.shape)
            # print('-----inputs shape:',inputs.shape)
            # print('-----labels shape:', labels.shape)
            # print(labels)
        
            loss = loss_func.get(mask,inputs,labels,black_box_func)
            # running_loss += loss.data[0]
            running_loss += loss.item()
            running_loss_train += loss.item()*inputs.size(0)

            # if(i%10 == 0):
            #     print('Epoch = %f , Loss = %f '%(epoch+1 , running_loss/(batch_size*(i+1))) )
        
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_loss_train = running_loss_train / len(trainloader.dataset)
        print("{} RawLoss: {:.4f} Loss: {:.4f}".format("train", epoch_loss,epoch_loss_train))
        save_checkpoint(net, checkpoint_file)

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize", type=int, default=8)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--areaL", type=float, default=8)
    parser.add_argument("--smoothL", type=float, default=0.5)
    parser.add_argument("--preserverL", type=float, default=0.3)
    parser.add_argument("--areaPowerL", type=float, default=0.3)
    # parser.add_argument("--checkpointName",type=str,default='data/checkpoints/s)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = args.batchSize
    num_workers = args.numWorkers
    regularizers = {'area_loss_coef': args.areaL, 'smoothness_loss_coef': args.smoothL, 'preserver_loss_coef': args.preserverL, 'area_loss_power': args.areaPowerL}
    checkpoint_path = os.path.join('data/checkpoints','saliency_model_test.tar')
    train(batch_size, num_workers, regularizers, device, checkpoint_path)
    
__main__()

