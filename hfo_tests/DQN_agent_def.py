#!/usr/bin/env python3
from __future__ import print_function

import argparse
import datetime
import itertools
import os
import pickle
import random

import numpy as np
# import matplotlib.pyplot as plt
from scipy.spatial import distance

from HFO_DQN import *
from hfo_utils import *

try:
    import hfo
except ImportError:
    print('Failed to import hfo. To install hfo, in the HFO directory'
          ' run: \"pip install .\"')
    exit()


params = {'SHT_DST': 0.136664020547, 'SHT_ANG': -0.747394386098,
          'PASS_ANG': 0.464086704478, 'DRIB_DST': -0.999052871962}


def get_ball_dist(state):
    agent = (state[0], state[1])
    ball = (state[3], state[4])
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
    parser.add_argument('--genmem', type=str, default=1,
                        help="generate memory to train")
    parser.add_argument('--train', type=str, default='True',
                        help="Train agent")
    args = parser.parse_args()
    gen_mem = int(args.genmem)
    training = int(args.train)
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
    rewards = [400, 1000]
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
    if gen_mem:
        for episode in itertools.count():
            status = hfo.IN_GAME
            done = True
            while status == hfo.IN_GAME:
                state = hfo_env.getState()
                state = remake_state(
                    state, num_teammates, num_opponents, is_offensive=False)
                if get_ball_dist(state) < 20:
                    action = random.randrange(0, 2)
                    hfo_env.act(actions[action])
                else:
                    action = 0
                    hfo_env.act(actions[0])
                # ------------------------------
                status = hfo_env.step()
                if status != hfo.IN_GAME:
                    done = 1
                else:
                    done = 0
                next_state = hfo_env.getState()
                next_state = remake_state(
                    next_state, num_teammates, num_opponents, is_offensive=False)
                # -----------------------------
                reward = 0
                if status == hfo.GOAL:
                    reward = -1000
                elif '-1' in hfo_env.statusToString(status):
                    reward = rewards[action]/3
                elif 'OUT' in hfo_env.statusToString(status):
                    reward = rewards[action]/2
                else:
                    if done:
                        reward = rewards[action]
                    else:
                        reward = rewards[action] - next_state[3]*3
                # Add experience to mem
                hfo_dqn.replay_buffer.push(
                    state, action, reward, next_state, done)
            # Quit if the server goes down
            if status == hfo.SERVER_DOWN:
                print('Saving memory')
                with open(saved_mem, 'wb') as f:
                    pickle.dump(hfo_dqn.replay_buffer, f)
                    f.close()
                hfo_env.act(hfo.QUIT)
                exit()
    else:
        if training:
            begin = datetime.datetime.now()
            episode_rewards = []
            epi_list = []
            for episode in itertools.count():
                status = hfo.IN_GAME
                frames = 1
                while status == hfo.IN_GAME:
                    state = hfo_env.getState()
                    state = remake_state(
                        state, num_teammates, num_opponents, is_offensive=False)
                    epsilon = hfo_dqn.epsilon_by_frame(frames)
                    action = hfo_dqn.act(state, epsilon)
                    if get_ball_dist(state) < 20:
                        # Where the magic happens
                        hfo_env.act(actions[action])
                    else:
                        hfo_env.act(actions[0])
                    frames += 1
                    # ------------------------------
                    status = hfo_env.step()
                    if status != hfo.IN_GAME:
                        done = 1
                    else:
                        done = 0
                    next_state = hfo_env.getState()
                    next_state = remake_state(
                        next_state, num_teammates, num_opponents, is_offensive=False)
                    # -----------------------------
                    reward = 0
                    if status == hfo.GOAL:
                        reward = -1000
                    elif '-1' in hfo_env.statusToString(status):
                        reward = rewards[action]/3
                    elif 'OUT' in hfo_env.statusToString(status):
                        reward = rewards[action]/2
                    else:
                        if done:
                            reward = rewards[action]
                        else:
                            reward = rewards[action] - next_state[3]*3
                    episode_rewards.append(reward)
                    # Add experience to mem
                    hfo_dqn.replay_buffer.push(
                        state, action, reward, next_state, done)
                    loss = compute_td_loss(hfo_dqn.batch_size, hfo_dqn)
                    losses.append(loss.item())
                    avg_losses.append((avg_loss + loss.item())/frames)
                    avg_loss = 0
                    try:
                        avg_loss = avg_loss + loss.item()
                    except:
                        pass
                    if done:
                        # Get the total reward of the episode
                        total_reward = np.sum(episode_rewards)/frames
                        epi_list.append('Episode: {} Total reward: {} Training loss: {:.4f}'.format(
                            episode, total_reward, avg_loss))
                        frames = 1
                        episode_rewards = []
                    if episode % 5 == 0:
                        torch.save(hfo_dqn.state_dict(), saved_model)

                # Quit if the server goes down
                if status == hfo.SERVER_DOWN:
                    end = datetime.datetime.now()
                    delta = end - begin
                    epi_file = open('{}k_training_rewards.txt'.format(10), 'w')
                    epi_file.writelines(epi_list)
                    epi_file.close()
                    print('End of Training: ', delta.seconds)
                    hfo_env.act(hfo.QUIT)
                    with open(saved_mem, 'wb') as f:
                        pickle.dump(hfo_dqn.replay_buffer, f)
                        f.close()
                    torch.save(hfo_dqn.state_dict(), saved_model)
                    exit()

        else:
            begin = datetime.datetime.now()
            epi_list = []
            episode_rewards = []
            for episode in itertools.count():
                status = hfo.IN_GAME
                frames = 1
                while status == hfo.IN_GAME:
                    state = hfo_env.getState()
                    state = remake_state(
                        state, num_teammates, num_opponents, False)
                    train_dqn = False
                    if get_ball_dist(state) < 20:
                        train_dqn = True
                        # Where the magic happens
                        epsilon = hfo_dqn.epsilon_by_frame(frames)
                        action = hfo_dqn.act(state, epsilon)
                        hfo_env.act(actions[action])
                    else:
                        action = 0
                        hfo_env.act(actions[0])
                    frames += 1
                    # ------------------------------
                    status = hfo_env.step()
                    if status != hfo.IN_GAME:
                        done = 1
                    else:
                        done = 0
                    next_state = hfo_env.getState()
                    next_state = remake_state(
                        next_state, num_teammates, num_opponents, False)
                    # -----------------------------
                    reward = 0
                    if status == hfo.GOAL:
                        reward = -20000
                    elif '-1' in hfo_env.statusToString(status):
                        reward = rewards[action]/4
                    elif 'OUT' in hfo_env.statusToString(status):
                        reward = rewards[action]/2
                    else:
                        if done:
                            reward = rewards[action]
                            if '-2' in hfo_env.statusToString(status):
                                reward = rewards[action]*2
                        else:
                            reward = rewards[action] - next_state[3]*3
                    episode_rewards.append(reward)
                    if done:
                        # We finished the episode
                        total_reward = np.sum(episode_rewards)/frames
                        epi_list.append('Episode: {} Total reward: {}'.format(
                            episode, total_reward))
                        episode_rewards = []
                        frames = 1
                    else:
                        episode_rewards.append(reward)
                    # ------------------------------------------------------- DOWN
                    # Quit if the server goes down
                    if status == hfo.SERVER_DOWN:
                        end = datetime.datetime.now()
                        delta = end - begin
                        epi_file = open('{}k_test_rewards.txt'.format(10), 'w')
                        epi_file.writelines(epi_list)
                        epi_file.close()
                        print('End of Training: ', delta.seconds)
                        hfo_env.act(hfo.QUIT)
                        exit()


if __name__ == '__main__':
    main()
