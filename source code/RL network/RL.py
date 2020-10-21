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

env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
action = namedtuple('action',
                        ('plus_half', 'plus_tenth', 'null', 'minus_tenth', 'minus_half'))
para = namedtuple('parameter',
                        ('sigma', 'beta_kasier'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class doubleDQN(nn.Module):

    def __init__(self, h, w, output1,output2):
        super(DQN, self).__init__()

        # image patch size is   h x w
        #first two  shared conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        # patch size after conv layer
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 64

        #  Parameter selection
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dense1 = nn.Linear(linear_input_size, 128)
        self.bn4 = nn.BatchNorm2d(128)
        self.head1 = nn.Linear(128,output1)

        linear_input_size = convw * convh * 128
        #  Parameter tuning
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.dense2 = nn.Linear(linear_input_size, 256)
        self.bn7 = nn.BatchNorm2d(256)
        self.head2 = nn.Linear(256, output2)

    
    def forward(self, x):

        # 2 shared layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

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










