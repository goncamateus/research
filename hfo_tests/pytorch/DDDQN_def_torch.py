import glob
import itertools
import math
import os
import sys
from collections import deque  # Ordered collection with ends
from pathlib import Path

import hfo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lib.Dueling_DQN_Torch import DuelingAgent as DQN_Agent
from lib.hfo_env import HFOEnv
from lib.hyperparameters import Config


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(DuelingDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs

        self.mlp1 = nn.Conv1d(
            self.input_shape[0], 32, kernel_size=3, bias=False)
        self.mlp2 = nn.Conv1d(32, 64, kernel_size=3, bias=False)
        self.mlp3 = nn.Conv1d(64, 64, kernel_size=3, bias=False)

        self.adv1 = nn.Linear(self.feature_size(), 512)
        self.adv2 = nn.Linear(512, self.num_actions)

        self.val1 = nn.Linear(self.feature_size(), 512)
        self.val2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        x = x.view(x.size(0), -1)
        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()

    def feature_size(self):
        x = self.mlp1(torch.zeros(1, *self.input_shape))
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x.view(1, -1).size(1)

    def sample_noise(self):
        # ignore this for now
        pass


class Model(DQN_Agent):
    def __init__(self, static_policy=False, env=None, config=None):
        self.stacked_frames = deque(
            [np.zeros(env.observation_space.shape, dtype=np.int)
             for i in range(16)], maxlen=16)
        super(Model, self).__init__(static_policy, env, config)
        self.num_feats = (*self.env.observation_space.shape,
                          len(self.stacked_frames))

    def declare_networks(self):
        self.model = DuelingDQN(
            (*self.env.observation_space.shape,
             len(self.stacked_frames)), self.env.action_space.n)
        self.target_model = DuelingDQN(
            (*self.env.observation_space.shape,
             len(self.stacked_frames)), self.env.action_space.n)

    def stack_frames(self, frame, is_new_episode):
        if is_new_episode:
            # Clear our stacked_frams
            self.stacked_frames = deque([np.zeros(
                frame.shape,
                dtype=np.int) for i in range(16)], maxlen=16)

            # Because we're in a new episode, copy the same frame 4x
            for _ in range(16):
                self.stacked_frames.append(frame)

            # Stack the frames
            stacked_state = np.stack(self.stacked_frames, axis=1)

        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different
            # frames)
            stacked_state = np.stack(self.stacked_frames, axis=1)

        return stacked_state


def main():
    config = Config()
    # epsilon variables
    config.epsilon_start = 1.0
    config.epsilon_final = 0.01
    config.epsilon_decay = 30000
    config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + \
        (config.epsilon_start - config.epsilon_final) * \
        math.exp(-1. * frame_idx / config.epsilon_decay)

    # misc agent variables
    config.GAMMA = 0.99
    config.LR = 1e-4
    # memory
    config.TARGET_NET_UPDATE_FREQ = 1000
    config.EXP_REPLAY_SIZE = 1000000
    config.BATCH_SIZE = 64
    config.PRIORITY_ALPHA = 0.6
    config.PRIORITY_BETA_START = 0.4
    config.PRIORITY_BETA_FRAMES = 100000

    # epsilon variables
    config.SIGMA_INIT = 0.5

    # Learning control variables
    config.LEARN_START = 1000000
    config.MAX_FRAMES = 60000000
    config.UPDATE_FREQ = 1

    # Nstep controls
    config.N_STEPS = 1

    actions = [hfo.MOVE, hfo.GO_TO_BALL]
    rewards = [700, 1000]
    hfo_env = HFOEnv(actions, rewards, strict=True)
    if hfo_env.getUnum() != 2:
        config.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    else:
        config.device = torch.device("cpu")

    log_dir = "/tmp/RC_test"
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    model = Model(env=hfo_env, config=config)

    model_path = './saved_agents/model_{}.dump'.format(hfo_env.getUnum())
    optim_path = './saved_agents/optim_{}.dump'.format(hfo_env.getUnum())
    mem_path = './saved_agents/exp_replay_agent_{}.dump'.format(
        hfo_env.getUnum())

    if os.path.isfile(model_path) and os.path.isfile(optim_path):
        model.load_w(model_path=model_path, optim_path=optim_path)
        print("Model Loaded")

    if os.path.isfile(mem_path):
        model.load_replay(mem_path=mem_path)
        model.learn_start = 0
        print("Memory Loaded")

    frame_idx = 1
    train = True
    gen_mem = False

    for episode in itertools.count():
        status = hfo.IN_GAME
        done = True
        episode_rewards = []
        while status == hfo.IN_GAME:
            if done:
                state = hfo_env.get_state(strict=True)
                frame = model.stack_frames(state, done)
            if train:
                if gen_mem and frame_idx / 4 < config.EXP_REPLAY_SIZE:
                    action = np.random.randint(0, model.env.action_space.n)
                else:
                    if gen_mem:
                        gen_mem = False
                        frame_idx = 0
                        model.learn_start = 0
                        print('Start Learning at Episode', episode)
                    epsilon = config.epsilon_by_frame(int(frame_idx / 4))
                    action = model.get_action(frame, epsilon, train=train)
            else:
                action = model.get_action(frame, 1.0, train=False)

            if hfo_env.get_ball_dist(state) > 20:
                action = 0

            next_state, reward, done, status = hfo_env.step(action,
                                                            strict=True)
            episode_rewards.append(reward)

            if done:
                # Get the total reward of the episode
                total_reward = np.sum(episode_rewards)
                model.finish_nstep()
                model.reset_hx()
                # We finished the episode
                next_state = np.zeros(state.shape)
                next_frame = np.zeros(frame.shape)
            else:
                next_frame = model.stack_frames(next_state, done)

            model.update(frame, action, reward,
                         next_frame, int(frame_idx / 4))
            frame = next_frame
            state = next_state

            frame_idx += 1
            if train:
                if int(frame_idx / 4) % 10000 == 0 and frame_idx > 5:
                    model.save_w(path_model=model_path,
                                 path_optim=optim_path)
                    print("Model Saved")
            if frame_idx / 4 > config.LEARN_START + 100 and frame_idx > 5:
                model.save_w(path_model=model_path, path_optim=optim_path)
                print("Model Saved")
                model.save_replay(mem_path=mem_path)
                print("Memory Saved")
                hfo_env.act(hfo.QUIT)
                exit()
        # Quit if the server goes down
        if status == hfo.SERVER_DOWN:
            model.save_w(path_model=model_path, path_optim=optim_path)
            print("Model Saved")
            model.save_replay(mem_path=mem_path)
            print("Memory Saved")
            hfo_env.act(hfo.QUIT)
            exit()


if __name__ == "__main__":
    main()
