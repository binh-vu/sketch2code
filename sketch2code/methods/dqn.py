#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Dict

import torch, random

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from collections import namedtuple

from sketch2code.methods.lstm import LSTM

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'not_done'))


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


class DQN(nn.Module):
    def __init__(self, img_h: int, img_w: int, dsl_vocab: Dict[int, str], dsl_embedding_dim: int, dsl_hidden_dim: int,
                 n_actions: int):
        super().__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.dsl_vocab = dsl_vocab
        self.dsl_hidden_dim = dsl_hidden_dim
        self.dsl_embedding_dim = dsl_embedding_dim
        self.n_actions = n_actions
        
        self.__build_model()
        
    def __build_model(self):
        # network compute features of target image
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(16, momentum=0.9)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32, momentum=0.9)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        imgsize = [self.img_w, self.img_h]
        for i, s in enumerate(imgsize):
            s = conv2d_size_out(s, 7, 2)
            s = pool2d_size_out(s, 3, 2)
            s = conv2d_size_out(s, 5, 1)
            s = pool2d_size_out(s, 3, 2)
            imgsize[i] = s
        linear_input_size = imgsize[0] * imgsize[1] * 32

        # compute features from programs
        self.lstm = LSTM(
            vocab_size=len(self.dsl_vocab),
            padding_token_idx=self.dsl_vocab['<pad>'],
            embedding_dim=self.dsl_hidden_dim,
            hidden_size=self.dsl_embedding_dim,
            n_layers=1)

        self.head = nn.Linear(linear_input_size + self.dsl_embedding_dim, self.n_actions)

    def forward(self, x1, x2, x2_lens):
        """
            @x1: desired images (batch: N x C x W x H)
            @x2: current programs (batch: N x T)
            @x2_lens: lengths of current programs (N)
        """
        batch_size = x2_lens.shape[0]
        
        x1 = self.pool1(F.selu(self.bn1(self.conv1(x1))))
        x1 = self.pool2(F.selu(self.bn2(self.conv2(x1))))
        
        # reshape from N x C x W x H to N x (C * W * H)
        x1 = x1.view(batch_size, -1)
        
        x2, (hn, cn) = self.lstm(x2, x2_lens, self.lstm.init_hidden(x2, batch_size))
        # flatten from N x T x H to N
        hn = hn.view(batch_size, -1)
        
        x = torch.cat((x1, hn), 1)
        return self.head(x)


def conv2d_size_out(size, kernel_size, stride):
    """
    Number of Linear input connections depends on output of conv2d layers
    and therefore the input image size, so compute it.
    """
    return (size - (kernel_size - 1) - 1) // stride + 1


def pool2d_size_out(size, kernel_size, stride, padding=0, dilation=1):
    return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1