#!/usr/bin/env python3
from __future__ import print_function

import argparse
import itertools
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np

from HFO_DQN import *

try:
    import hfo
except ImportError:
    print('Failed to import hfo. To install hfo, in the HFO directory'
          ' run: \"pip install .\"')
    exit()


params = {'SHT_DST': 0.136664020547, 'SHT_ANG': -0.747394386098,
          'PASS_ANG': 0.464086704478, 'DRIB_DST': -0.999052871962}


def can_shoot(goal_dist, goal_angle):
    """Returns True if if player may have a good shot at the goal"""
    return bool((goal_dist < params['SHT_DST']) and (goal_angle > params['SHT_ANG']))


def has_better_pos(dist_to_op, goal_angle, pass_angle, curr_goal_angle):
    """Returns True if teammate is in a better attacking position"""
    if (curr_goal_angle > goal_angle) or (dist_to_op < params['DRIB_DST']):
        return False
    if pass_angle < params['PASS_ANG']:
        return False
    return True


def get_action(state, hfo_env, num_teammates, rand_pass):
    """Decides and performs the action to be taken by the agent."""

    goal_dist = float(state[6])
    goal_op_angle = float(state[8])
    if can_shoot(goal_dist, goal_op_angle):
        hfo_env.act(hfo.SHOOT)
        return
    team_list = list(range(num_teammates))
    if rand_pass and (num_teammates > 1):
        random.shuffle(team_list)
    for i in team_list:
        teammate_uniform_number = state[10 + 3*num_teammates + 3*i + 2]
        if has_better_pos(dist_to_op=float(state[10 + num_teammates + i]),
                          goal_angle=float(state[10 + i]),
                          pass_angle=float(state[10 + 2*num_teammates + i]),
                          curr_goal_angle=goal_op_angle):
            hfo_env.act(hfo.PASS, teammate_uniform_number)
            return
    # no check for can_dribble is needed; doDribble in agent.cpp includes
    # (via doPreprocess) doForceKick, which will cover this situation since
    # existKickableOpponent is based on distance.
    hfo_env.act(hfo.DRIBBLE)
    return


def main():
    # ----------------------------------------------------CONECTION TO SERVER-----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6000, help="Server port")
    parser.add_argument('--seed', type=int, default=None,
                        help="Python randomization seed; uses python default if 0 or not given")
    parser.add_argument('--rand-pass', action="store_true",
                        help="Randomize order of checking teammates for a possible pass")
    parser.add_argument('--epsilon', type=float, default=0,
                        help="Probability of a random action if has the ball, to adjust difficulty")
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
                                'formations-dt', args.port,
                                'localhost', 'base_left', False,
                                record_dir=args.rdir)
    else:
        hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
                                'formations-dt', args.port,
                                'localhost', 'base_left', False)
# ----------------------------------------------------CONECTION TO SERVER-----------------------------------------
    num_teammates = hfo_env.getNumTeammates()
    num_opponents = hfo_env.getNumOpponents()
    if args.seed:
        if (args.rand_pass and (num_teammates > 1)) or (args.epsilon > 0):
            print("Python randomization seed: {0:d}".format(args.seed))
        else:
            print(
                "Python randomization seed useless without --rand-pass w/2+ teammates or --epsilon >0")
    if args.rand_pass and (num_teammates > 1):
        print("Randomizing order of checking for a pass")
    epsilon = 1.0
    if args.epsilon > 0:
        print("Using epsilon {0:n}".format(args.epsilon))
        epsilon = args.epsilon


# ----------------------------------------------------GOT PARAMS----------------------------------------------------
    actions = [hfo.SHOOT, hfo.PASS, hfo.DRIBBLE, hfo.GO_TO_BALL, hfo.NOOP]
    rewards = [1000, 700, 450, 300, 10]
    device = torch.device("cuda:0")
    hfo_dqn = DQN(13 + 3*num_teammates, len(actions), epsilon)
    saved_mem = 'SAVED_MEMORY.mem'
    saved_model = 'SAVED_MODEL.pt'
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
            dqn_state = np.append(state[:10+3*num_teammates+1], state[-2:])
            action = 4
            if int(state[5]) == 1:  # state[5] is 1 when player has the ball
                # Where the magic happens
                epsilon = hfo_dqn.epsilon_by_frame(frames)
                frames += 1
                action = hfo_dqn.act(dqn_state, epsilon)
                if action == 1:
                    num_had_ball += 1
                    goal_op_angle = float(state[8])
                    team_list = list(range(num_teammates))
                    for i in team_list:
                        teammate_uniform_number = state[10 +
                                                        3*num_teammates + 3*i + 2]
                        if has_better_pos(dist_to_op=float(state[10 + num_teammates + i]),
                                          goal_angle=float(state[10 + i]),
                                          pass_angle=float(
                                              state[10 + 2*num_teammates + i]),
                                          curr_goal_angle=goal_op_angle):
                            hfo_env.act(hfo.PASS, teammate_uniform_number)
                else:
                    hfo_env.act(actions[action])
                num_eps += 1
            else:
                hfo_env.act(hfo.MOVE)
                num_move += 1
            status = hfo_env.step()
            if status != hfo.IN_GAME:
                done = 1
            else:
                done = 0
            next_state = hfo_env.getState()
            dqn_next_state = np.append(
                next_state[:10+3*num_teammates+1], next_state[-2:])
            assert action < 5 and action >= 0, 'action = {}'.format(action)
            reward = 0
            if int(next_state[5]) == 1:
                if status == hfo.GOAL:
                    reward = 1000
                else:
                    if done:
                        reward = -rewards[action]/15
                    elif action != 0:
                        reward = rewards[action]
                    else:
                        reward = 0
            hfo_dqn.replay_buffer.push(
                dqn_state, action, reward, dqn_next_state, done)
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

        if args.epsilon > 0:
            print("\tNum move: {0:d}; Random action: {1:d}; Nonrandom: {2:d}".format(num_move,
                                                                                     num_eps,
                                                                                     (num_had_ball -
                                                                                      num_eps)))


if __name__ == '__main__':
    main()
