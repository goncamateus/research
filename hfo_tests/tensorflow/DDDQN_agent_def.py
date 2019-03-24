#!/usr/bin/env python3
from __future__ import print_function

import argparse
import datetime
import itertools
import os
import pickle
import random

import numpy as np
import tensorflow as tf  # Deep Learning library
# import matplotlib.pyplot as plt
from scipy.spatial import distance

from Dueling_Double_DQN import (DDDQNNet_MLP, Memory, Params, SumTree,
                                predict_action, stack_frames,
                                update_target_graph)
from hfo_utils import remake_state, strict_state

try:
    import hfo
except ImportError:
    print('Failed to import hfo. To install hfo, in the HFO directory'
          ' run: \"pip install .\"')
    exit()


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
    mem_size = 1000000
    if 'memories/memory_{}_{}vs{}_def_{}.mem'.format(hfo_env.getUnum(), num_teammates, num_opponents, mem_size) in os.listdir('./memories'):
        with open('memories/memory_{}_{}vs{}_def_{}.mem'.format(hfo_env.getUnum(), num_teammates, num_opponents, mem_size), 'rb') as memfile:
            memory = pickle.load(memfile)
            memfile.close()
    else:
        memory = Memory(mem_size)
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
                    state = remake_state(
                        state, num_teammates, num_opponents, is_offensive=False)
                    if done:
                        frame, parametros.stacked_frames = stack_frames(
                            parametros.stacked_frames, state, True, state.shape, len(actions))
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
                    with open('memories/memory_{}_{}vs{}_def_{}.mem'.format(hfo_env.getUnum(), num_teammates, num_opponents, mem_size), 'wb') as f:
                        pickle.dump(memory, f)
                        f.close()
                    hfo_env.act(hfo.QUIT)
                    exit()
    except KeyboardInterrupt:
        print('Saving memory due to Interrupt. Episodes: ', ep)
        with open('memories/memory_{}_{}vs{}_def_{}.mem'.format(hfo_env.getUnum(), num_teammates, num_opponents, mem_size), 'wb') as f:
            pickle.dump(memory, f)
            f.close()
# ----------------------------------------------Generate Memory----------------------------------------------------------
    else:
        # Instantiate the DQNetwork
        DQNetwork = DDDQNNet_MLP([21, 4], parametros.action_size,
                                 parametros.learning_rate, name="DQNetwork")

        # Instantiate the target network
        TargetNetwork = DDDQNNet_MLP([21, 4], parametros.action_size,
                                     parametros.learning_rate, name="TargetNetwork")
        # Losses
        tf.summary.scalar("Loss", DQNetwork.loss)
        write_op = tf.summary.merge_all()
        loss = None
        # Saver will help us to save our model
        saver = tf.train.Saver()
        try:
            if parametros.training:
                epi_list = []
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
                            state = remake_state(
                                state, num_teammates, num_opponents, is_offensive=False)
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
                                act = action.index(1)
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
                            next_state = remake_state(
                                next_state, num_teammates, num_opponents, is_offensive=False)
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
                                    reward = rewards[act] - next_state[3]*3
                            episode_rewards.append(reward)
                            if done:
                                # Get the total reward of the episode
                                total_reward = np.sum(episode_rewards)
                                # epi_list.append('Episode: {} Total reward: {} Training loss: {:.4f} Explore P: {:.4f}'.format(
                                #     episode, total_reward, loss, explore_probability))
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
                            states_mb = np.array([each[0][0]
                                                  for each in batch], ndmin=2)
                            actions_mb = np.array(
                                [each[0][1] for each in batch])
                            rewards_mb = np.array(
                                [each[0][2] for each in batch])
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
                        if episode % 5 == 0:
                            save_path = saver.save(sess, "./models/model_{}_{}vs{}_def.ckpt".format(
                                hfo_env.getUnum(), num_teammates, num_opponents))
                            print("Model Saved")
                        # ------------------------------------------------------- DOWN
                        # Quit if the server goes down
                        if status == hfo.SERVER_DOWN:
                            epi_file = open(
                                '{}k_training_rewards.txt'.format(int(parametros.total_episodes/1000)), 'w')
                            epi_file.writelines(epi_list)
                            epi_file.close()
                            print('Saving memory')
                            with open('memories/memory_{}_{}vs{}_def_{}.mem'.format(hfo_env.getUnum(), num_teammates, num_opponents, mem_size), 'wb') as f:
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
                    epi_list = []
                    # Load the model
                    saver.restore(sess, "./models/model_2_1vs1_def.ckpt")
                    episode_rewards = []
                    for episode in itertools.count():
                        status = hfo.IN_GAME
                        done = True
                        while status == hfo.IN_GAME:
                            state = hfo_env.getState()
                            state = remake_state(
                                state, num_teammates, num_opponents, False)
                            state = strict_state(state, 1, 1,
                                               num_teammates, num_opponents)
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
                                action = random.choice(
                                    parametros.possible_actions)

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
                            next_state = remake_state(
                                next_state, num_teammates, num_opponents, False)
                            next_state = strict_state(next_state, 1, 1,
                                               num_teammates, num_opponents)
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
                                    reward = rewards[act] - next_state[3]*3
                            if done:
                                # We finished the episode
                                next_frame = np.zeros(frame.shape)
                                total_reward = np.sum(episode_rewards)
                                epi_list.append('Episode: {} Total reward: {}'.format(
                                    episode, total_reward))
                            else:
                                # Get the next state
                                next_frame, parametros.stacked_frames = stack_frames(
                                    parametros.stacked_frames, next_state, False, state.shape, len(actions))
                                frame = next_frame
                                episode_rewards.append(reward)
                        # ------------------------------------------------------- DOWN
                        # Quit if the server goes down
                        if status == hfo.SERVER_DOWN:
                            hfo_env.act(hfo.QUIT)
                            saver.save(sess, './exported/my_model')
                            tf.train.write_graph(sess.graph, '.', "./exported/graph.pb", as_text=False)
                            tf.train.write_graph(sess.graph, '.', "./exported/graph.pb_txt", as_text=True)
                            epi_file = open(
                                '{}k_test_rewards.txt'.format(int(parametros.total_episodes/1000)), 'w')
                            epi_file.writelines(epi_list)
                            epi_file.close()
                            exit()
        except KeyboardInterrupt:
            save_path = saver.save(sess, "./models/model_{}_{}vs{}_def.ckpt".format(
                hfo_env.getUnum(), num_teammates, num_opponents))
            print("Model Saved")


if __name__ == '__main__':
    main()
