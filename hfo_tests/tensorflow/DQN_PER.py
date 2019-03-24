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


class DQNetwork:
    """MLP DQN"""

    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, x, 4]
            self.inputs_ = tf.placeholder(
                tf.float32, [None, *state_size], name="inputs")
                
            self.actions_ = tf.placeholder(
                tf.float32, [None, action_size], name="actions_")

            self.ISWeights_ = tf.placeholder(
                tf.float32, [None, 1], name='IS_weights')

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            MLP
            BatchNormalization
            ELU
            """
            self.mlp1 = tf.layers.dense(inputs=self.inputs_,
                                        units=32,
                                        activation=tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        name="mlp1")

            self.mlp1_batchnorm = tf.layers.batch_normalization(self.mlp1,
                                                                training=True,
                                                                epsilon=1e-5,
                                                                name='batch_norm1')

            self.mlp1_out = tf.nn.elu(self.mlp1_batchnorm, name="mlp1_out")

            """
            Second convnet:
            MLP
            BatchNormalization
            ELU
            """
            self.mlp2 = tf.layers.dense(inputs=self.inputs_,
                                        units=64,
                                        activation=tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        name="mlp2")

            self.mlp2_batchnorm = tf.layers.batch_normalization(self.mlp2,
                                                                training=True,
                                                                epsilon=1e-5,
                                                                name='batch_norm2')

            self.mlp2_out = tf.nn.elu(self.mlp2_batchnorm, name="mlp2_out")

            """
            Third convnet:
            MLP
            BatchNormalization
            ELU
            """
            self.mlp3 = tf.layers.dense(inputs=self.inputs_,
                                        units=128,
                                        activation=tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        name="mlp3")

            self.mlp3_batchnorm = tf.layers.batch_normalization(self.mlp3,
                                                                training=True,
                                                                epsilon=1e-5,
                                                                name='batch_norm3')

            self.mlp3_out = tf.nn.elu(self.mlp3_batchnorm, name="mlp3_out")

            self.flatten = tf.layers.flatten(self.mlp3_out)

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size,
                                          activation=None)

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


class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0

    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """

    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        """
        Here we add our priority score in the sumtree leaf and add the experience in data
        """

        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        # If we're above the capacity, you go back to first index (we overwrite)
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    """
    Update the leaf priority score and propagate the change through tree
    """

    def update(self, tree_index, priority):
        """"""
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:    # this method is faster than the recursive loop in the reference code

            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 

            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to
    #  avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking
    #  only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)

    def store(self, experience):
        """ Store a new experience in our tree
        Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)"""
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since
        # this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)   # set the max p for new p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty(
            (n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min(
            [1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(
            self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(
                n * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index
            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities on the tree
    """

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


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
