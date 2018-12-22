from collections import namedtuple

import gym_super_mario_bros
import numpy as np
import torch
import torchvision.transforms as T
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    next_state, reward, done, info = env.step(env.action_space.sample())
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))
    screen_width = screen[1].shape[1]
    x = info['x_pos']
    init = x - 80
    if init < 0:
        init = 0
    end = init + 160
    if end > screen_width:
        end = screen_width
        init = screen_width - 160
    screen = screen[:, 80:-30, init:end]
    screen = screen.transpose((1, 2, 0))
    screen = np.dot(screen[...,:3], [0.299, 0.587, 0.114])
env.close()
