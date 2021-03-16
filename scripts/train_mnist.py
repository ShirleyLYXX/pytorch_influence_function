#! /usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import time
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_data(batch_size=500):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    train_images = trainset.data
    train_labels = trainset.targets
    perm = np.load(os.path.join("data", "small_mnist_idx.npy"))
    train_images = train_images[perm, :]
    train_labels = train_labels[perm]

    trainset.data = train_images
    trainset.targets = train_labels

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader



class all_CNN_c(nn.Module):
    def __init__(self, num_classes=10, input_channels=1, hidden_units=8, conv_patch_size=3):
        super(all_CNN_c, self).__init__()
        self.conv1_a = nn.Conv2d(in_channels=input_channels, out_channels=hidden_units, kernel_size=conv_patch_size, stride=1)
        self.conv1_c = nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=conv_patch_size, stride=2)

        self.conv2_a = nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=conv_patch_size, stride=1)
        self.conv2_c = nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=conv_patch_size, stride=2)

        self.conv3_a = nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=conv_patch_size, stride=1)
        last_layer_units = 10
        self.conv3_c = nn.Conv2d(in_channels=hidden_units, out_channels=last_layer_units, kernel_size=1, stride=1)

        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(last_layer_units, num_classes)


    def forward(self, x):
        x = F.tanh(self.conv1_a(x))
        x = F.tanh(self.conv1_c(x))
        x = F.tanh(self.conv2_a(x))
        x = F.tanh(self.conv2_c(x))
        x = F.tanh(self.conv3_a(x))
        x = F.tanh(self.conv3_c(x))
        x = self.pool(x)

        x = x.view(-1, 10 * 1 * 1)
        x = self.fc1(x)
        return x

def test_acc(testloader, net):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            #images, labels = data
            images, labels = data[0].cuda(), data[1].cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            _, pred = torch.max(outputs, 1)
            c = (pred == labels).squeeze()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

def save_model(net, PATH):
    torch.save(net.state_dict(), PATH)


def load_model(PATH):
    net = all_CNN_c()
    net.load_state_dict(torch.load(PATH))
    net.cuda()
    return net


def train(trainloader, testloader, net, epochs=100000, learning_rate=0.0001, weight_decay=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000, 20000], gamma=0.1)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = data
            start_time = time.time()
            inputs, labels = data[0].cuda(), data[1].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print statistics
            running_loss += loss.item()

        if epoch % 100 == 99:
            print('[%d, %5d] loss: %.3f (%.3f secs)' %
                      (epoch, i, running_loss / (len(trainloader)), time.time()-start_time))

        if epoch % 2000 == 1999:
            test_acc(testloader, net)
            save_model(net, PATH='./output/small_mnist_all_cnn_c_%d.pth' % epoch)

    print('Finished Training')


if __name__ == "__main__":
    trainloader, testloader = load_data()
    model = all_CNN_c()
    model.cuda()
    
    num_steps = 2000
    train(trainloader, testloader, model, epochs=num_steps, learning_rate=0.0001, weight_decay=0.001)
