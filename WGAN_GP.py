#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn

latent_size = 128

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        convT1 = nn.Sequential(
            # in: latent_size x 1 x 1
            nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
            # out: 512 x 4 x 4
        )
        
        convT2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
            # out: 256 x 8 x 8
        )
        
        convT3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
            # out: 128 x 16 x 16
        )
        
        convT4 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # out: 3 x 32 x 32
        )
        
        self.convT1 = convT1
        self.convT2 = convT2
        self.convT3 = convT3
        self.convT4 = convT4
        
    def forward(self, input):
        output = self.convT1(input)
        output = self.convT2(output)
        output = self.convT3(output)
        output = self.convT4(output)
        
        return output
    

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        conv1 = nn.Sequential(
            # in: 3 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
            # out: 32 x 16 x 16
        )
        
        conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 8 x 8
        )
        
        conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
            # out: 128 x 4 x 4
        )
        
        conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 1 x 1

            nn.Flatten()
        )
        
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.linear = nn.Linear(256, 1)
        
    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.linear(output)
        return output