#!/usr/bin/env python3
from __future__ import print_function

import argparse
import itertools
import os
import pickle
import random
import datetime

# import matplotlib.pyplot as plt
from scipy.spatial import distance

from HFO_DQN import *

try:
    import hfo
except ImportError:
    print('Failed to import hfo. To install hfo, in the HFO directory'
          ' run: \"pip install .\"')
    exit()


params = {'SHT_DST': 0.136664020547, 'SHT_ANG': -0.747394386098,
          'PASS_ANG': 0.464086704478, 'DRIB_DST': -0.999052871962}


def get_ball_dist(state):
    agent = (state[0]*43, state[1]*43)
    ball = (state[3]*43, state[4]*43)
    return distance.euclidean(agent, ball)


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
    args = parser.parse_args()
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
    actions = [hfo.MOVE, hfo.GO_TO_BALL]
    rewards = [100, 1000]
    device = torch.device("cuda:0")
    hfo_dqn = DQN(hfo_env.getStateSize(), len(actions), epsilon)
    saved_mem = 'SAVED_MEMORY_{}def_{}_{}.mem'.format(
        hfo_env.getUnum(), num_teammates, num_opponents)
    saved_model = 'SAVED_MODEL_{}def_{}_{}.pt'.format(
        hfo_env.getUnum(), num_teammates, num_opponents)
    if saved_model in os.listdir('.'):
        hfo_dqn.load_state_dict(torch.load(saved_model))
    if saved_mem in os.listdir('.'):
        with open(saved_mem, 'rb') as in_file:
            mem = pickle.load(in_file)
            hfo_dqn.replay_buffer = mem
            in_file.close()
    hfo_dqn.to(device)
    losses = []
    avg_loss = 0
    avg_losses = []
# ----------------------------------------------DQN PARAMS----------------------------------------------------------
    begin = datetime.datetime.now()
    for episode in itertools.count():
        status = hfo.IN_GAME
        frames = 1
        while status == hfo.IN_GAME:
            state = hfo_env.getState()

            train_dqn = False
            print(get_ball_dist(state))
            if get_ball_dist(state) < 15:
                train_dqn = True
                # Where the magic happens
                epsilon = hfo_dqn.epsilon_by_frame(frames)
                frames += 1
                action = hfo_dqn.act(state, epsilon)
                hfo_env.act(actions[action])
            else:
                hfo_env.act(actions[0])
            status = hfo_env.step()
            if status != hfo.IN_GAME:
                done = 1
            else:
                done = 0
            if train_dqn:
                next_state = hfo_env.getState()
                reward = 0
                if int(next_state[5]) == 1:
                    if status == hfo.GOAL:
                        reward = -1000
                    elif '-1' in hfo_env.statusToString(status):
                        reward = rewards[action]/2
                    else:
                        if done:
                            reward = rewards[action]
                        else:
                            reward = rewards[action] - state[0]*43*3
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
            end = datetime.datetime.now()
            delta = end - begin
            print('End of Training: ', delta.seconds)
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



if __name__ == '__main__':
    main()
