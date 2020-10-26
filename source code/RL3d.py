import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Bm3d



# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class doubleDQN3d(nn.Module):

    def __init__(self, h, w, output1,output2):
        super(doubleDQN3d, self).__init__()

        # image patch size is   h x w x d
        #first two  shared conv layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=(2),padding=(1,1,1))  # 5x5x3
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=(2),padding=(1,1,0))  # 3x3x1
        self.bn2 = nn.BatchNorm3d(64)

        # patch size after conv layer
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return size // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))

        linear_input_size = convw * convh * 64

        
        #  Parameter selection
        #  conv 64 Filter, kernel 3x3 
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        #  FCL 128 neurons
        self.dense1 = nn.Linear(linear_input_size, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.head1 = nn.Linear(128,output1)
       
        linear_input_size = convw * convh  * 128
        #  Parameter tuning
        #  conv 128 Filter, kernel 3x3 
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        #  conv 128 Filter, kernel 3x3 
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        #  FCL 256 neurons
        self.dense2 = nn.Linear(linear_input_size, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.head2 = nn.Linear(256, output2)

        self.opt = optim.SGD(self.parameters(), lr=0.00001, momentum=0.9)

    
    def forward(self, x):


        # 2 shared layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # reshape
        x = x.view(x.size(0),x.size(1),x.size(2),x.size(3))
        
        #  Parameter selection
        out1 = F.relu(self.bn3(self.conv3(x)))
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.bn4(self.dense1(out1)))
        out1 = self.head1(out1)
       
        #  Parameter tuning
        out2 = F.relu(self.bn5(self.conv4(x)))
        out2 = F.relu(self.bn6(self.conv5(out2)))
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.bn7(self.dense2(out2)))
        out2 = self.head2(out2)

        return  out1, out2


    def update(self, x,target1,target2):

        out1,out2 = self.forward(x)
        self.opt.zero_grad()

        loss1 = nn.MSELoss(reduction='sum')
        loss2 = nn.MSELoss(reduction='sum')
        upd_1 = loss1(out1,target1 )
        upd_2 = loss2(out2,target2 )
        upd_ = upd_1 + upd_2
        upd_.backward()
        
        self.opt.step()

"""
if __name__ == '__main__':

    model = doubleDQN3d(9,9,2,5)
    batch = torch.randn(4, 1, 9,9,5)
    output1, output2 = model(batch)
"""