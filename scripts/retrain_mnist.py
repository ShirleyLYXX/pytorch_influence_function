# coding:utf-8
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import time
import math
from train_mnist import load_data, load_model, save_model
import numpy as np
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


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


def frozen_retrain(net, trainloader, testloader, epochs=100, learning_rate=0.0001):
    start_time_scatch = time.time()
    criterion = nn.CrossEntropyLoss()
    freeze_params = [p for p in net.parameters() if p.requires_grad][0:-2]
    for p in freeze_params:
        p.requires_grad = False
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

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

            # print statistics
            running_loss += loss.item()

        if epoch % 10 == 9:
            print('[%d, %5d] loss: %.8f (%.3f secs)' %
                      (epoch, i, running_loss / (len(trainloader)), time.time()-start_time))

        if epoch % 100 == 99:
            test_acc(testloader, net)

    print("Finished Training: %.3f (secs)" % (time.time() - start_time_scatch))

    return net

def retrain(net, trainloader, testloader, epochs=100, learning_rate=0.0001):
    start_time_scatch = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

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

            # print statistics
            running_loss += loss.item()

        if epoch % 10 == 9:
            print('[%d, %5d] loss: %.8f (%.3f secs)' %
                      (epoch, i, running_loss / (len(trainloader)), time.time()-start_time))

        if epoch % 100 == 99:
            test_acc(testloader, net)

    print("Finished Training: %.3f (secs)" % (time.time() - start_time_scatch))

    return net


def get_loss_on_test(testloader, test_id, net):
    criterion = nn.CrossEntropyLoss()

    # get the inputs; data is a list of [inputs, labels]
    z_test, t_test = testloader.dataset[test_id]
    z_test = testloader.collate_fn([z_test])
    t_test = testloader.collate_fn([t_test])

    z_test, t_test = z_test.cuda(), t_test.cuda()

    net.eval()
    outputs = net(z_test)
    loss = criterion(outputs, t_test)

    return loss


def fill_feed_dict_with_all_but_one_ex(idx_to_remove, batch_size=500, input_size=28):
    train_loader, test_loader = load_data()
    trainset = train_loader.dataset
    train_images = trainset.data
    train_labels = trainset.targets

    train_images = train_images[torch.arange(train_images.size(0))!=idx_to_remove]
    train_labels = train_labels[torch.arange(train_labels.size(0))!=idx_to_remove]

    train_loader.dataset.data = train_images
    train_loader.dataset.targets = train_labels

    return train_loader

#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    # Read influence.json
    test_id = 6558
    model_path = "output/small_mnist_all_cnn_c_99999.pth"
    frozen = True

    import json
    dict_ = json.load(open("outdir/influence_results_single_%d_1.json" % test_id))
    predicted_loss_diffs = np.array(dict_[str(test_id)]['influence'])
    num_to_remove = 100
    indices_to_remove = np.argsort(np.abs(predicted_loss_diffs))[-num_to_remove:]
    predicted_loss_diffs = predicted_loss_diffs[indices_to_remove]
    actual_loss_diffs = np.zeros([num_to_remove])

    # prepare nets and data
    model = load_model(model_path)
    train_loader, test_loader = load_data()
    #---------------------------------
    # Sanity check
    test_loss_val = get_loss_on_test(test_loader, test_id, model)
    print("======= Retraining with all train data =======")

    if frozen:
        model = frozen_retrain(model, train_loader, test_loader, epochs=100, learning_rate=0.000001)
    else:
        model = retrain(model, train_loader, test_loader, epochs=100, learning_rate=0.000001)

    retrained_test_loss_val = get_loss_on_test(test_loader, test_id, model)
    
    print('Sanity check: what happens if you train the model a bit more?')
    print('Loss on test idx with original model    : %s' % test_loss_val)
    print('Loss on test idx with retrained model   : %s' % retrained_test_loss_val)
    print('Difference in test loss after retraining     : %s' % (retrained_test_loss_val - test_loss_val))
    print('===')
    print('These differences should be close to 0.\n')


    # Retraining experiment
    for counter, idx_to_remove in enumerate(indices_to_remove):
        print("=== #%s ===" % counter)
        print('Retraining without train_idx %s :' % (idx_to_remove))
        #print('Retraining without train_idx %s (label %s):' % (idx_to_remove, train_loader.dataset[idx_to_remove]['label']))

        # Restore params
        model = load_model(model_path)
        retrain_loader = fill_feed_dict_with_all_but_one_ex(idx_to_remove)
        print(len(retrain_loader.dataset))

        if frozen:
            model = frozen_retrain(model, retrain_loader, test_loader, epochs=100, learning_rate=0.000001)
        else:
            model = retrain(model, retrain_loader, test_loader, epochs=100, learning_rate=0.000001) # retrain

        retrained_test_loss_val = get_loss_on_test(test_loader, test_id, model)
        actual_loss_diffs[counter] = retrained_test_loss_val - test_loss_val

        # print('Diff in params: %s' % np.linalg.norm(np.concatenate(params_val) - np.concatenate(retrained_params_val)))      
        print('Loss on test idx with original model    : %s' % test_loss_val)
        print('Loss on test idx with retrained model   : %s' % retrained_test_loss_val)
        print('Difference in loss after retraining     : %s' % actual_loss_diffs[counter])
        print('Predicted difference in loss (influence): %s' % predicted_loss_diffs[counter])

    np.savez(
        'outdir/loss_diffs_small_mnist_all_cnn_c_99999_retrain-100-finetune-p2.npz',
        actual_loss_diffs=actual_loss_diffs, 
        predicted_loss_diffs=predicted_loss_diffs)
    
    #print('Correlation is %s' % pearsonr(actual_loss_diffs, predicted_loss_diffs)[0])
