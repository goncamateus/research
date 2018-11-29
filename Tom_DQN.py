import math
import random
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from IPython.display import clear_output

np.random.seed(53)
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


epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500


def epsilon_by_frame(frame_idx): return epsilon_final + (epsilon_start -
                                                         epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, num_actions)
        )
        

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            # adds extra dim at front
            state = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action = q_value.max(1)[1]
        else:
            action = random.randrange(env.action_space.n)
        return action


env = gym.make('MountainCar-v0')

model = DQN(env.observation_space.shape[0], env.action_space.n)

model = model.cuda()

optimizer = optim.Adam(model.parameters())


def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(state))
    next_state = Variable(torch.FloatTensor(next_state))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


num_frames = 1600
batch_size = 32
# 2 possible actions each round, batch_size rounds per batch
replay_buffer = ReplayBuffer(2 * batch_size)
gamma = 0.99
losses = []
episode_reward = 0
avg_loss = 0
avg_losses = []
state = env.reset()
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(state, epsilon)
    next_state, reward, done, _ = env.step(int(action))
    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward
    env.render()
    if done:
        state = env.reset()

    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size)
        losses.append(loss.item())
        avg_losses.append((avg_loss + loss.item())/frame_idx)
        avg_loss = 0
    try:
        avg_loss = avg_loss + loss.item()
    except:
        pass

env.close()
