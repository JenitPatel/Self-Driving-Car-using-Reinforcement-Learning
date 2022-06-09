# AI for Automatic Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


# Neural Network Design

class Neural(nn.Module):

    def __init__(self, parameter_size, action_nn):
        super(Neural, self).__init__()
        self.parameter_size = parameter_size
        self.action_nn = action_nn
        self.p1 = nn.Linear(parameter_size, 30)
        self.p2 = nn.Linear(30, action_nn)

    def forward(self, state):
        x = F.relu(self.p1(state))
        Qval = self.p2(x)
        return Qval


# Implementing recap memory

class RecapMemory(object):

    def __init__(self, max_amount):
        self.max_amount = max_amount
        self.recall = []

    def memory_add(self, event):
        self.recall.append(event)
        if len(self.recall) > self.max_amount:
            del self.recall[0]

    def instance(self, batch_size):
        instances = zip(*random.sample(self.recall, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), instances)


# Deep Q Network Approach

class DeepQNetwork():

    def __init__(self, parameter_size, action_nn, gamma):
        self.gamma = gamma
        self.prize_window = []
        self.model = Neural(parameter_size, action_nn)
        self.recall = RecapMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(parameter_size).unsqueeze(0)
        self.last_action = 0
        self.last_prize = 0

    def decision_make(self, state):
        probability = F.softmax(self.model(Variable(state, volatile=True)) * 100)  # T=100
        # action = probs.multinomial(0)
        # replacing  1 is the number of samples to draw
        action = probability.multinomial(1)

        return action.data[0, 0]

    def gain_info(self, batch_state, batch_next_state, batch_prize, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_prize
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        #  td_loss.backward(retain_variables = True)
        # Replaceing
        td_loss.backward(retain_graph=True)

        self.optimizer.step()

    def update(self, prize, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.recall.memory_add(
            (self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_prize])))
        action = self.decision_make(new_state)
        if len(self.recall.recall) > 100:
            batch_state, batch_next_state, batch_action, batch_prize = self.recall.instance(100)
            self.gain_info(batch_state, batch_next_state, batch_prize, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_prize = prize
        self.prize_window.append(prize)
        if len(self.prize_window) > 1000:
            del self.prize_window[0]
        return action

    def score(self):
        return sum(self.prize_window) / (len(self.prize_window) + 1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading controlpoint... ")
            controlpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(controlpoint['state_dict'])
            self.optimizer.load_state_dict(controlpoint['optimizer'])
            print("done !")
        else:
            print("no controlpoint found...")