import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from scipy import misc
from resnet import *

batch_size = 32
num_workers = 4
num_epochs = 15

def save_checkpoint(state, filename='black_box_func.tar'):
    torch.save(state, filename)

def load_checkpoint(net,optimizer,filename='black_box_func.tar'):
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return net,optimizer


def cifar10():
    
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
 

  
def train():
    trainloader,testloader,classes = cifar10()
    model_name = 'alexnet'
    black_box_func = BlackBoxModel(model_name = model_name, pretrained=True, num_classes=10)
    black_box_func.toCuda()
    black_box_func = black_box_func.getModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(black_box_func.parameters())

    for epoch in range(num_epochs):  # loop over the dataset multiple times
    print("----- Epoch {}/{}".format(epoch, num_epochs))
    running_loss = 0.0
    running_corrects = 0.0
    for i, data in enumerate(trainloader, 0):
        # print('trainLoader size: ',data)
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        out = black_box_func(inputs)
        # print('out: ',out)
        # print('labels: ',labels)
        _, preds = torch.max(out.data, 1)
        # print('preds: ', preds)
        # print('labels: ',labels.data)
        loss = criterion(out,labels)   
        running_corrects += torch.sum(preds == labels.data)
        # print('running_corrects: ', running_corrects)
        # print('denominator: ', str(batch_size*(i+1)))
        # running_loss += loss.data[0]
        running_loss += loss.item()
        
        # if(i%100 == 0):
        #   print('Epoch = %f , Accuracy = %f, Loss = %f '%(epoch+1 , running_corrects/(batch_size*(i+1)), running_loss/(batch_size*(i+1))) )
        
        loss.backward()
        optimizer.step()
    
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = running_corrects.double() / len(trainloader.dataset)
    print("{} Loss: {:.4f} Acc: {:.4f}".format('train', epoch_loss, epoch_acc))

    save_checkpoint(black_box_func, filename='data/checkpoints/black_box_model_' + str(model_name) + '.tar')

  def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliencyModel", type=str, default='data/checkpoints/saliency_model_2222.tar')
    args = parser.parse_args()
    saliencyModel = args.saliencyModel
    visualize(saliency_model=saliencyModel)
    
__main__()