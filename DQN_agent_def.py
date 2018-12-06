#!/usr/bin/env python3
from __future__ import print_function

import argparse
import itertools
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from HFO_DQN import *

try:
    import hfo
except ImportError:
    print('Failed to import hfo. To install hfo, in the HFO directory'
          ' run: \"pip install .\"')
    exit()


params = {'SHT_DST': 0.136664020547, 'SHT_ANG': -0.747394386098,
          'PASS_ANG': 0.464086704478, 'DRIB_DST': -0.999052871962}

def main():
    # ----------------------------------------------------CONECTION TO SERVER-----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6000, help="Server port")
    parser.add_argument('--seed', type=int, default=None,
                        help="Python randomization seed; uses python default if 0 or not given")
    parser.add_argument('--epsilon', type=float, default=0,
                        help="Probability of a random action, to adjust difficulty")
    parser.add_argument('--record', action='store_true',
                        help="If doing HFO --record")
    parser.add_argument('--rdir', type=str, default='log/',
                        help="Set directory to use if doing --record")
    args=parser.parse_args()
    if args.seed:
        random.seed(args.seed)
    hfo_env = hfo.HFOEnvironment()
    if args.record:
        hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
                            './formations-dt', args.port,
                            'localhost', 'base_right', play_goalie=False,
                            record_dir=args.rdir)
    else:
        hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
                            './formations-dt', args.port,
                            'localhost', 'base_right', play_goalie=False)
# ----------------------------------------------------CONECTION TO SERVER-----------------------------------------
    num_teammates = hfo_env.getNumTeammates()
    num_opponents = hfo_env.getNumOpponents()
    if args.seed:
        if (args.rand_pass and (num_teammates > 1)) or (args.epsilon > 0):
            print("Python randomization seed: {0:d}".format(args.seed))
        else:
            print(
                "Python randomization seed useless without --rand-pass w/2+ teammates or --epsilon >0")

    epsilon = 1.0
    if args.epsilon > 0:
        print("Using epsilon {0:n}".format(args.epsilon))
        epsilon = args.epsilon


# ----------------------------------------------------GOT PARAMS----------------------------------------------------
    actions = [hfo.DEFEND_GOAL, hfo.MARK_PLAYER, hfo.GO_TO_BALL]
    rewards = [600, 700, 1000]
    device = torch.device("cuda:0")
    hfo_dqn = DQN(hfo_env.getStateSize(), len(actions), epsilon)
    saved_mem = 'SAVED_MEMORY_{}def_{}_{}.mem'.format(hfo_env.getUnum(),num_teammates,num_opponents)
    saved_model = 'SAVED_MODEL_{}def_{}_{}.pt'.format(hfo_env.getUnum(),num_teammates,num_opponents)
    if saved_model in os.listdir('.'):
        hfo_dqn.load_state_dict(torch.load(saved_model))
    if saved_mem in os.listdir('.'):
        with open(saved_mem, 'rb') as f:
            mem = pickle.load(f)
            hfo_dqn.replay_buffer = mem
            f.close()
    hfo_dqn.to(device)
    losses = []
    episode_reward = 0
    avg_loss = 0
    avg_losses = []
# ----------------------------------------------DQN PARAMS----------------------------------------------------------
    for episode in itertools.count():
        num_eps = 0
        num_had_ball = 0
        num_move = 0
        status = hfo.IN_GAME
        frames = 1
        while status == hfo.IN_GAME:
            state = hfo_env.getState()
            action = 2
            if int(state[5]) != 1:  # state[5] is 1 when player has the ball
                # Where the magic happens
                epsilon = hfo_dqn.epsilon_by_frame(frames)
                frames += 1
                action = hfo_dqn.act(state, epsilon)
                closest_op_unum = state[12 + 6*num_teammates]
                if action == 1:
                    hfo_env.act(hfo.MARK_PLAYER, closest_op_unum)
                else:
                    hfo_env.act(actions[action])
            else:
                hfo_env.act(hfo.MOVE)
            status = hfo_env.step()
            if status != hfo.IN_GAME:
                done = 1
            else:
                done = 0
            next_state = hfo_env.getState()
            assert action < 5 and action >= 0, 'action = {}'.format(action)
            reward = 0
            if int(next_state[5]) == 1:
                if status == hfo.GOAL:
                    reward = -1000
                else:
                    if done:
                        reward = rewards[action]
                    else:
                        reward = rewards[action] - state[0]*5
            hfo_dqn.replay_buffer.push(
                state, action, reward, next_state, done)
            if len(hfo_dqn.replay_buffer) > hfo_dqn.batch_size:
                loss = compute_td_loss(hfo_dqn.batch_size, hfo_dqn)
                losses.append(loss.item())
                avg_losses.append((avg_loss + loss.item())/frames)
                avg_loss = 0
            try:
                avg_loss = avg_loss + loss.item()
            except:
                pass

        # Quit if the server goes down
        if status == hfo.SERVER_DOWN:
            hfo_env.act(hfo.QUIT)
            # plt.figure(figsize=(20,5))
            # plt.plot(avg_losses)
            # plt.show()
            exit()

        # Check the outcome of the episode
        print("Episode {0:d} ended with {1:s}".format(episode,
                                                      hfo_env.statusToString(status)))
        with open(saved_mem, 'wb') as f:
            pickle.dump(hfo_dqn.replay_buffer, f)
            f.close()
        torch.save(hfo_dqn.state_dict(), saved_model)

        # if args.epsilon > 0:
        #     print("\tNum move: {0:d}; Random action: {1:d}; Nonrandom: {2:d}".format(num_move,
        #                                                                              num_eps,
        #                                                                              (num_had_ball -
        #                                                                               num_eps)))


if __name__ == '__main__':
    main()
