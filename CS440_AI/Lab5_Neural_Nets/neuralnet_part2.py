# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP5. You should only modify code
within this file and neuralnet_part1.py,neuralnet_leaderboard -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.cnn1 = nn.Sequential(nn.Conv2d(3,6, kernel_size=3), nn.ReLU())
        self.pool = nn.MaxPool2d(2,2)
        self.cnn2 = nn.Sequential(nn.Conv2d(6,16, kernel_size=3), nn.ReLU())
        self.nerualNetwork = nn.Sequential(nn.Linear(576, 64),nn.ReLU(),nn.Linear(64, 8),nn.ReLU(),nn.Linear(8,out_size))
        self.lrate = lrate
        self.optims = optim.SGD(self.parameters(), self.lrate,momentum=0.9,weight_decay=0.004)
        
    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        N = x.shape[0]
        x = x.view(N,3,31,31)
        x = self.pool(self.cnn1(x))
        #print(x.shape)
        x =self.cnn2(x)
        x = self.pool(x)
        #print(x.shape)
        x = torch.flatten(x,1)
        #print(x.shape)
        return self.nerualNetwork(x)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        self.optims.zero_grad()
        lossFunction = self.loss_fn(self.forward(x), y)
        lossFunction.backward()
        self.optims.step()

        return lossFunction.item()
def preprocess(data_set):
    for i,data in enumerate(data_set.data):
        data_set.data[i] = data / np.linalg.norm(data)
        #print(data)
def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    loss_fn = nn.CrossEntropyLoss()
    lrate = 0.0002

    net = NeuralNet(lrate, loss_fn, len(train_set[0]), 4)
    losses = []
    yhats = []
    data = get_dataset_from_arrays(train_set, train_labels)
    #preprocess(data)
    #print(data.data.shape)
    trainloader = DataLoader(data,batch_size=batch_size,shuffle=False)

    for _ in range(epochs):            
        for data in trainloader:
            #print(data)
            inputs = data['features']
            labels = data['labels']
            #print(inputs.shape,labels.shape)
            net.optims.zero_grad()
            steps = net.step(inputs,labels)
            net.optims.step()
        losses.append(steps)
    #preprocess(dev_set)
    #print(dev_set)
    network = net(dev_set).detach().numpy()
    yhats = np.argmax(network,axis=1).astype(int)
    return losses, yhats, net