import gym
import gym_super_mario_bros
import numpy as np
from gym import spaces
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)


def get_state(state, info):
    s = state.transpose((2, 0, 1))
    s_width = s[1].shape[1]
    x = info['x_pos']
    init = x - 80
    if init < 0:
        init = 0
    end = init + 160
    if end > s_width:
        end = s_width
        init = s_width - 160
    s = s[:, 80:-30, init:end]
    s = s.transpose((1, 2, 0))
    s = np.dot(s[..., :3], [0.299, 0.587, 0.114])
    return s
