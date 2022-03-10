import numpy as np

import torch
from torch import nn

class Actor(nn.Module):
    def __init__(self, num_states, num_actions, hidden1=400, hidden2=300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class Critic(nn.Module):
    def __init__(self, num_states, num_actions, hidden1=400, hidden2=300):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_states, hidden1)
        self.fc2 = nn.Linear(hidden1+num_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self,xa):
        x,a = xa
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out