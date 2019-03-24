import os
import pickle
import random  # Handling random number generation
import time  # Handling time calculation
from collections import deque  # Ordered collection with ends

import numpy as np  # Handle matrices
import tensorflow as tf  # Deep Learning library


class Params():
    """ This do shit"""

    # MODEL HYPERPARAMETERS
    action_size = 2           # n possible actions
    possible_actions = None
    learning_rate = 0.00025      # Alpha (aka learning rate)

    # TRAINING HYPERPARAMETERS
    total_episodes = 50000         # Total episodes for training
    max_steps = 5000              # Max possible steps in an episode
    batch_size = 64

    # FIXED Q TARGETS HYPERPARAMETERS
    max_tau = 10000  # Tau is the C step where we update our target network

    # EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start
    explore_stop = 0.01            # minimum exploration probability
    decay_rate = 0.00005            # exponential decay rate for exploration prob

    # Q LEARNING hyperparameters
    gamma = 0.95               # Discounting rate

    # MEMORY HYPERPARAMETERS
    # If you have balls change to 1million
    # Number of experiences stored in the Memory when initialized for the first time
    pretrain_length = 1000000
    memory_size = 1000000       # Number of experiences the Memory can keep

    # We stack 4 frames
    stack_size = 4

    # Initialize deque with zero-images one array for each image
    stacked_frames = None

    def __init__(self, state_shape, action_size, training=True, gen_mem=False):
        self.action_size = action_size
        self.possible_actions = np.identity(action_size, dtype=int).tolist()
        self.stacked_frames = deque([np.zeros(state_shape, dtype=np.int)
                                     for i in range(self.stack_size)], maxlen=4)
        self.training = training
        self.gen_mem = gen_mem


def stack_frames(stacked_frams, frame, is_new_episode, state_shape, actions):
    """Preprocess frame"""

    if is_new_episode:
        # Clear our stacked_frams
        stacked_frams = Params(state_shape, actions).stacked_frames

        # Because we're in a new episode, copy the same frame 4x
        stacked_frams.append(frame)
        stacked_frams.append(frame)
        stacked_frams.append(frame)
        stacked_frams.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frams, axis=1)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frams.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frams, axis=1)

    return stacked_state, stacked_frams


class DDDQNNet:
    """CNN DQN"""

    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):

            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 160,130, 4]
            self.inputs_ = tf.placeholder(
                tf.float32, [None, *state_size], name="inputs")

            #
            self.ISWeights_ = tf.placeholder(
                tf.float32, [None, 1], name='IS_weights')

            self.actions_ = tf.placeholder(
                tf.float32, [None, action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.Q = tf.placeholder(tf.float32, [None], name="output")

            """
            First convnet:
            CNN
            ELU
            """
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=128,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.layers.flatten(self.conv3_out)

            # Here we separate into two streams
            # The one that calculate V(s)
            self.value_fc = tf.layers.dense(inputs=self.flatten,
                                            units=512,
                                            activation=tf.nn.elu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="value_fc")

            self.value = tf.layers.dense(inputs=self.value_fc,
                                         units=1,
                                         activation=None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name="value")

            # The one that calculate A(s,a)
            self.advantage_fc = tf.layers.dense(inputs=self.flatten,
                                                units=512,
                                                activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="advantage_fc")

            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                             units=self.action_size,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="advantages")

            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + \
                tf.subtract(self.advantage, tf.reduce_mean(
                    self.advantage, axis=1, keepdims=True))

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(
                self.output, self.actions_), axis=1)

            # The loss is modified because of PER
            self.absolute_errors = tf.abs(
                self.target_Q - self.Q)  # for updating Sumtree

            self.loss = tf.reduce_mean(
                self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss)


class DDDQNNet_MLP:
    """MLP DQN"""

    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):

            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, x, 4]
            self.inputs_ = tf.placeholder(
                tf.float32, [None, *state_size], name="inputs")

            #
            self.ISWeights_ = tf.placeholder(
                tf.float32, [None, 1], name='IS_weights')

            self.actions_ = tf.placeholder(
                tf.float32, [None, action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First net:
            MLP
            ELU
            """
            self.layer1 = tf.layers.dense(inputs=self.inputs_,
                                          units=32,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name="mlp1")

            """
            Second net:
            MLP
            ELU
            """
            self.layer2 = tf.layers.dense(inputs=self.layer1,
                                          units=64,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name="mlp2")

            """
            Third net:
            MLP
            ELU
            """
            self.layer3 = tf.layers.dense(inputs=self.layer2,
                                          units=128,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name="mlp3")

            self.flatten = tf.layers.flatten(self.layer3)

            # Here we separate into two streams
            # The one that calculate V(s)
            self.value_fc = tf.layers.dense(inputs=self.flatten,
                                            units=128,
                                            activation=tf.nn.elu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="value_fc")

            self.value = tf.layers.dense(inputs=self.value_fc,
                                         units=1,
                                         activation=None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name="value")

            # The one that calculate A(s,a)
            self.advantage_fc = tf.layers.dense(inputs=self.flatten,
                                                units=128,
                                                activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="advantage_fc")

            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                             units=self.action_size,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="advantages")

            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + \
                tf.subtract(self.advantage, tf.reduce_mean(
                    self.advantage, axis=1, keepdims=True))

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(
                self.output, self.actions_), axis=1)

            # The loss is modified because of PER
            self.absolute_errors = tf.abs(
                self.target_Q - self.Q)  # for updating Sumtree

            self.loss = tf.reduce_mean(
                self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss)


"""
This function will do the part
With ϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
"""


def predict_action(sess, DQNetwork, explore_start, explore_stop, decay_rate, decay_step, state, possible_actions):
    # EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    # First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + \
        (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(possible_actions)

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict={
                      DQNetwork.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability

# This function helps us to copy one set of variables to another
# In our case we use it when we want to copy the parameters of DQN to Target_network
# Thanks of the very good implementation of Arthur Juliani https://github.com/awjuliani


def update_target_graph():

    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

    # Get the parameters of our Target_network
    to_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder
