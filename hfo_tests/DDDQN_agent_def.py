#!/usr/bin/env python
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

import tensorflow as tf  # Deep Learning library
from Dueling_Double_DQN import *

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
    mates = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    memory = [Memory(100000) for x in os.listdir('./memories')]
    for i, x in enumerate(os.listdir('./memories')):
        x = './memories/' + x
        with open(x, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            memory[i] = unpickler.load()
            f.close()
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
    try:
        memory = memory[mates.index(hfo_env.getUnum())]
    except:
        memory = Memory(100000)
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

    gen_mem = int(args.genmem)
# ----------------------------------------------------GOT PARAMS------------------------------------------------
    actions = [hfo.MOVE, hfo.GO_TO_BALL]
    rewards = [400, 1000]
    parametros = Params(hfo_env.getStateSize(),
                        len(actions), training=int(args.train))
    # Reset the graph
    tf.reset_default_graph()

# ----------------------------------------------DDDQN parametros----------------------------------------------------------
    try:
        ep = 0
        if gen_mem:
            for episode in itertools.count():
                ep = episode
                status = hfo.IN_GAME
                done = True
                while status == hfo.IN_GAME:
                    state = hfo_env.getState()
                    if done:
                        frame, parametros.stacked_frames = stack_frames(
                            parametros.stacked_frames, state, True, state.shape, len(actions))
                    if get_ball_dist(state) < 15:
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
                            reward = rewards[action] - next_state[0]*43*3
                    p = [0 for x in range(len(actions))]
                    p[action] = 1
                    action = np.array(p)
                    if done:
                        # We finished the episode
                        next_frame = np.zeros(frame.shape)
                        experience = frame, action, reward, next_frame, done
                        memory.store(experience)
                    else:
                        # Get the next state
                        next_frame, parametros.stacked_frames = stack_frames(
                            parametros.stacked_frames, next_state, False, state.shape, len(actions))

                        # Add experience to mem
                        experience = frame, action, reward, next_frame, done
                        memory.store(experience)
                        frame = next_frame
                # Quit if the server goes down
                if status == hfo.SERVER_DOWN:
                    print('Saving memory')
                    with open('memories/memory_{}_{}vs{}_def.mem'.format(hfo_env.getUnum(), num_teammates, num_opponents), 'wb') as f:
                        pickle.dump(memory, f)
                        f.close()
                    hfo_env.act(hfo.QUIT)
                    exit()
    except KeyboardInterrupt:
        print('Saving memory due to Interrupt. Episodes: ', ep)
        with open('memories/memory_{}_{}vs{}_def.mem'.format(hfo_env.getUnum(), num_teammates, num_opponents), 'wb') as f:
            pickle.dump(memory, f)
            f.close()
# ----------------------------------------------Generate Memory----------------------------------------------------------
    else:
        # Instantiate the DQNetwork
        DQNetwork = DDDQNNet_MLP([hfo_env.getStateSize(), 4], parametros.action_size,
                                 parametros.learning_rate, name="DQNetwork")

        # Instantiate the target network
        TargetNetwork = DDDQNNet_MLP([hfo_env.getStateSize(), 4], parametros.action_size,
                                     parametros.learning_rate, name="TargetNetwork")
        # Losses
        tf.summary.scalar("Loss", DQNetwork.loss)
        write_op = tf.summary.merge_all()

        # Saver will help us to save our model
        saver = tf.train.Saver()
        try:
            if parametros.training:
                with tf.Session() as sess:
                    if "model_{}_{}vs{}_def.ckpt".format(hfo_env.getUnum(), num_teammates, num_opponents) in os.listdir('./models'):
                        # Load the model
                        saver.restore(sess, "./models/model_{}_{}vs{}_def.ckpt".format(
                            hfo_env.getUnum(), num_teammates, num_opponents))
                    else:
                        sess.run(tf.global_variables_initializer())
                    # Initialize the decay rate (that will use to reduce epsilon)
                    decay_step = 0

                    # Set tau = 0
                    tau = 0
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = update_target_graph()
                    sess.run(update_target)
                    nmr_out = 0
                    taken = 0
                    for episode in itertools.count():
                        status = hfo.IN_GAME
                        done = True
                        while status == hfo.IN_GAME:
                            state = hfo_env.getState()
                            if done:
                                # Initialize the rewards of the episode
                                episode_rewards = []
                                frame, parametros.stacked_frames = stack_frames(
                                    parametros.stacked_frames, state, True, state.shape, len(actions))

                            # Increase the C step
                            tau += 1

                            # Increase decay_step
                            decay_step += 1

                            # With ϵ select a random action atat, otherwise select a = argmaxQ(st,a)
                            action, explore_probability = predict_action(sess, DQNetwork,
                                                                         parametros.explore_start, parametros.explore_stop, parametros.decay_rate,
                                                                         decay_step, frame, parametros.possible_actions)

                            if get_ball_dist(state) < 20:
                                hfo_env.act(actions[action.index(1)])
                            else:
                                act = 0
                                action = np.array([1, 0])
                                hfo_env.act(actions[0])
                            # ------------------------------
                            status = hfo_env.step()
                            if status != hfo.IN_GAME:
                                done = 1
                            else:
                                done = 0
                            next_state = hfo_env.getState()
                            # -----------------------------
                            reward = 0
                            if status == hfo.GOAL:
                                reward = -20000
                            elif '-1' in hfo_env.statusToString(status):
                                reward = rewards[act]/4
                            elif 'OUT' in hfo_env.statusToString(status):
                                nmr_out += 1
                                reward = rewards[act]/2
                                if nmr_out % 20 and nmr_out > 1:
                                    reward = reward*10
                            else:
                                if done:
                                    reward = rewards[act]
                                    if '-2' in hfo_env.statusToString(status):
                                        taken += 1
                                        reward = rewards[act]*2
                                        if taken % 5 and taken > 1:
                                            reward = reward*20
                                else:
                                    reward = rewards[act] - next_state[0]*43*3
                            episode_rewards.append(reward)
                            if done:
                                # Get the total reward of the episode
                                total_reward = np.sum(episode_rewards)
                                print('Episode: {}'.format(episode),
                                      'Total reward: {}'.format(total_reward),
                                      'Training loss: {:.4f}'.format(loss),
                                      'Explore P: {:.4f}'.format(explore_probability))
                                # We finished the episode
                                next_frame = np.zeros(frame.shape)
                                experience = frame, action, reward, next_frame, done
                                memory.store(experience)
                            else:
                                # Get the next state
                                next_frame, parametros.stacked_frames = stack_frames(
                                    parametros.stacked_frames, next_state, False, state.shape, len(actions))

                                # Add experience to mem
                                experience = frame, action, reward, next_frame, done
                                memory.store(experience)
                                frame = next_frame


                            # LEARNING PART
                            # Obtain random mini-batch from memory
                            tree_idx, batch, ISWeights_mb = memory.sample(
                                parametros.batch_size)
                            states_mb = np.array([each[0][0] for each in batch], ndmin=2)
                            actions_mb = np.array([each[0][1] for each in batch])
                            rewards_mb = np.array([each[0][2] for each in batch])
                            next_states_mb = np.array([each[0][3]
                                                    for each in batch], ndmin=2)
                            dones_mb = np.array([each[0][4] for each in batch])

                            target_Qs_batch = []

                            # DOUBLE DQN Logic
                            # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                            # Use TargetNetwork to calculate the Q_val of Q(s',a')

                            # Get Q values for next_state
                            q_next_state = sess.run(DQNetwork.output, feed_dict={
                                                    DQNetwork.inputs_: next_states_mb})

                            # Calculate Qtarget for all actions that state
                            q_target_next_state = sess.run(TargetNetwork.output, feed_dict={
                                TargetNetwork.inputs_: next_states_mb})

                            # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a')
                            for i in range(0, len(batch)):
                                terminal = dones_mb[i]

                                # We got a'
                                action = np.argmax(q_next_state[i])

                                # If we are in a terminal state, only equals reward
                                if terminal:
                                    target_Qs_batch.append(rewards_mb[i])

                                else:
                                    # Take the Qtarget for action a'
                                    target = rewards_mb[i] + parametros.gamma * \
                                        q_target_next_state[i][action]
                                    target_Qs_batch.append(target)

                            targets_mb = np.array(
                                [each for each in target_Qs_batch])
                            _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                                                feed_dict={DQNetwork.inputs_: states_mb,
                                                                           DQNetwork.target_Q: targets_mb,
                                                                           DQNetwork.actions_: actions_mb,
                                                                           DQNetwork.ISWeights_: ISWeights_mb})

                            # Update priority
                            memory.batch_update(tree_idx, absolute_errors)

                            # Write TF Summaries
                            summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                                    DQNetwork.target_Q: targets_mb,
                                                                    DQNetwork.actions_: actions_mb,
                                                                    DQNetwork.ISWeights_: ISWeights_mb})
                            if tau > parametros.max_tau:
                                # Update the parameters of our TargetNetwork with DQN_weights
                                update_target = update_target_graph()
                                sess.run(update_target)
                                tau = 0
                                print("Model updated")
                        if episode%5 ==0:
                            save_path = saver.save(sess, "./models/model_{}_{}vs{}_def.ckpt".format(
                            hfo_env.getUnum(), num_teammates, num_opponents))
                            print("Model Saved")
                        #------------------------------------------------------- DOWN
                        # Quit if the server goes down
                        if status == hfo.SERVER_DOWN:
                            print('Saving memory')
                            with open('memories/memory_{}_{}vs{}_def.mem'.format(hfo_env.getUnum(), num_teammates, num_opponents), 'wb') as f:
                                pickle.dump(memory, f)
                                f.close()
                            save_path = saver.save(sess, "./models/model_{}_{}vs{}_def.ckpt".format(
                                hfo_env.getUnum(), num_teammates, num_opponents))
                            print("Model Saved")
                            hfo_env.act(hfo.QUIT)
                            exit()
            # # WHEN TESTING
            else:
                with tf.Session() as sess:

                    # Load the model
                    saver.restore(sess, "./models/model_{}_{}vs{}_def.ckpt".format(
                            hfo_env.getUnum(), num_teammates, num_opponents))

                    for episode in itertools.count():
                        status = hfo.IN_GAME
                        done = True
                        while status == hfo.IN_GAME:
                            state = hfo_env.getState()
                            if done:
                                # Initialize the rewards of the episode
                                episode_rewards = []
                                frame, parametros.stacked_frames = stack_frames(
                                    parametros.stacked_frames, state, True, state.shape, len(actions))

                            # EPSILON GREEDY STRATEGY
                            # Choose action a from state s using epsilon greedy.
                            # First we randomize a number
                            exp_exp_tradeoff = np.random.rand()

                            explore_probability = 0.01

                            if (explore_probability > exp_exp_tradeoff):
                                # Make a random action (exploration)
                                action = random.choice(parametros.possible_actions)

                            else:
                                # Get action from Q-network (exploitation)
                                # Estimate the Qs values state
                                Qs = sess.run(DQNetwork.output, feed_dict={
                                            DQNetwork.inputs_: frame.reshape((1, *frame.shape))})

                                # Take the biggest Q value (= the best action)
                                act = np.argmax(Qs)
                                action = parametros.possible_actions[int(act)]
                            if get_ball_dist(state) < 20:
                                hfo_env.act(actions[action.index(1)])
                            else:
                                act = 0
                                action = np.array([1, 0])
                                hfo_env.act(actions[0])
                            # ------------------------------
                            status = hfo_env.step()
                            if status != hfo.IN_GAME:
                                done = 1
                            else:
                                done = 0
                            next_state = hfo_env.getState()
                            # -----------------------------
                            reward = 0
                            if status == hfo.GOAL:
                                reward = -20000
                            elif '-1' in hfo_env.statusToString(status):
                                reward = rewards[act]/4
                            elif 'OUT' in hfo_env.statusToString(status):
                                reward = rewards[act]/2
                            else:
                                if done:
                                    reward = rewards[act]
                                    if '-2' in hfo_env.statusToString(status):
                                        reward = rewards[act]*2
                                else:
                                    reward = rewards[act] - next_state[0]*43*3
                            if done:
                                # We finished the episode
                                next_frame = np.zeros(frame.shape)
                                experience = frame, action, reward, next_frame, done
                                memory.store(experience)
                            else:
                                # Get the next state
                                next_frame, parametros.stacked_frames = stack_frames(
                                    parametros.stacked_frames, next_state, False, state.shape, len(actions))

                                # Add experience to mem
                                experience = frame, action, reward, next_frame, done
                                memory.store(experience)
                                frame = next_frame
                        #------------------------------------------------------- DOWN
                        # Quit if the server goes down
                        if status == hfo.SERVER_DOWN:
                            print('Saving memory')
                            with open('memories/memory_{}_{}vs{}_def.mem'.format(hfo_env.getUnum(), num_teammates, num_opponents), 'wb') as f:
                                pickle.dump(memory, f)
                                f.close()
                            hfo_env.act(hfo.QUIT)
                            exit()
        except KeyboardInterrupt:
            print('Saving memory')
            with open('memories/memory_{}_{}vs{}_def.mem'.format(hfo_env.getUnum(), num_teammates, num_opponents), 'wb') as f:
                pickle.dump(memory, f)
                f.close()
            save_path = saver.save(sess, "./models/model_{}_{}vs{}_def.ckpt".format(
                hfo_env.getUnum(), num_teammates, num_opponents))
            print("Model Saved")


if __name__ == '__main__':
    main()
