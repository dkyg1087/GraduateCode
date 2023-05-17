import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
    
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 128, kernel_size = 4, stride=2, padding=1,bias=False)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride=2, padding=1,bias=False)
        self.batchn1 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride=2, padding=1,bias=False)
        self.batchn2 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 4, stride=2, padding=1,bias=False)
        self.batchn3 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(in_channels = 1024, out_channels = 1, kernel_size = 4, stride=1, padding=1,bias=False)
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = F.leaky_relu(self.conv1(x),negative_slope = 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(self.batchn1(x),negative_slope = 0.2)
        x = self.conv3(x)
        x = F.leaky_relu(self.batchn2(x),negative_slope = 0.2)
        x = self.conv4(x)
        x = F.leaky_relu(self.batchn3(x),negative_slope = 0.2)
        x = self.conv5(x)
        
        ##########       END      ##########
        
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################

        self.convt1=nn.ConvTranspose2d(in_channels=noise_dim, out_channels=1024, kernel_size=4, stride=1, padding=0,bias=False)
        self.batchn1 = nn.BatchNorm2d(1024)
        self.convt2=nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1,bias=False)
        self.batchn2 = nn.BatchNorm2d(512)
        self.convt3=nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1,bias=False)
        self.batchn3 = nn.BatchNorm2d(256)
        self.convt4=nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1,bias=False)
        self.batchn4 = nn.BatchNorm2d(128)
        self.convt5=nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1,bias=False)
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################

        x = self.convt1(x)
        x = F.relu(self.batchn1(x))
        x = self.convt2(x)
        x = F.relu(self.batchn2(x))
        x = self.convt3(x)
        x = F.relu(self.batchn3(x))
        x = self.convt4(x)
        x = F.relu(self.batchn4(x))
        x = self.convt5(x)
        x = F.tanh(x)
        
        ##########       END      ##########
        return x
    

