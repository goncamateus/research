import math
import random
from collections import deque
from itertools import count

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, epsilon_start=1.0):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.num_actions = num_actions
        self.optimizer = optim.Adam(self.parameters())
        self.batch_size = 512
        self.replay_buffer = ReplayBuffer(num_actions * self.batch_size)
        self.gamma = 0.99
        self.epsilon_start = epsilon_start
        self.epsilon_final = 0.01
        self.epsilon_decay = 500

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            # adds extra dim at front
            state = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action = q_value.max(1)[1]
        else:
            action = random.randrange(self.num_actions)
        return action

    def epsilon_by_frame(self, frame_idx): return self.epsilon_final + (self.epsilon_start -
                                                                        self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)


def compute_td_loss(batch_size, model):
    state, action, reward, next_state, done = model.replay_buffer.sample(
        batch_size)

    state = Variable(torch.FloatTensor(state))
    next_state = Variable(torch.FloatTensor(next_state))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + model.gamma * next_q_value * (1 - done)
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()

    return loss
