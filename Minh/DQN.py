from collections import deque
import random
import numpy as np
import math
import tensorflow as tf
from SinglePacketRoutingEnv import SinglePacketRoutingEnv
from LinkFailureEnv import LinkFailureEnv
from TrafficEnv import TrafficEnv
import os
import matplotlib.pyplot as plt
import gym

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MAX_EPSILON = 1
MIN_EPSILON = 0.01

LAMBDA = 0.0005


class DDQN(object):

    def __init__(self, env=None, seed=None):
        nodes1 = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"]
        links1 = [["n1", "n2", 1], ["n1", "n0", 1], ["n0", "n2", 1], ["n3", "n0", 1], ["n2", "n5", 2], ["n3", "n5", 1],
                  ["n4", "n3", 1], ["n4", "n6", 10], ["n6", "n3", 1], ["n4", "n7", 1], ["n6", "n7", 1], ["n5", "n6", 1]]
        self.env = SinglePacketRoutingEnv(nodes1, links1)
        #self.env = gym.make('CartPole-v0')
        #self.env = TrafficEnv(nodes1, links1)
        #self.env = LinkFailureEnv(nodes1, links1, 0)

        self.batch_size = 128
        self.memory_capacity = 10000
        self.memory = ExperienceReplay(max_length=self.memory_capacity)
        self.memory_required = self.batch_size * 3
        self.epsilon = MAX_EPSILON
        self.learning_rate_start = 0.001
        self.regularization_scale = 0.01
        self.gamma = 0.999

        self.all_rewards = []
        self.obs_no, = self.env.observation_space.shape
        #print("Observation space shape is" + str(self.env.observation_space.shape))
        #print("Observation no is" + str(self.obs_no))
        self.action_no = self.env.action_space.n

        self.episode_number = 3500
        self.activation_function = "tanh"
        self.initializer = "he_init"
        self.dropout_rate = 0.0

        tf.reset_default_graph()
        self.observation_format = tf.placeholder(dtype=tf.float32, shape=[self.obs_no, 1])
        self.action_format = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.reward_format = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.global_step = tf.Variable(0, trainable=False)

        self.learning_rate = tf.train.exponential_decay(self.learning_rate_start, self.global_step, 40000, 0.01)

        self.q_network = self.create_network(tensor=self.observation_format, activation=self.activation_function,
                                             initializer=self.initializer, name="q_network")
        self.target_network = self.create_network(tensor=self.observation_format, activation=self.activation_function,
                                             initializer=self.initializer, name="target_network")

        self.q_network_sum = tf.reduce_sum(self.q_network * tf.one_hot(self.action_format, self.action_no), axis=1)
        self.target_network_sum = tf.reduce_sum(self.target_network * tf.one_hot(self.action_format, self.action_no),
                                                 axis=1)

        self.q_network_loss = tf.square(self.reward_format - self.q_network_sum)
        self.target_network_loss = tf.square(self.reward_format - self.target_network_sum)

        # Initializing optimizers
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_q_network = optimizer.minimize(self.q_network_loss, global_step=self.global_step)
        self.train_target_network = optimizer.minimize(self.target_network_loss)

        self.learn(self.episode_number)

    def create_network(self, tensor, initializer, activation, name):

        #print("Placeholder is " + str(tensor))

        l1_regularizer = tf.contrib.layers.l1_regularizer(self.regularization_scale)
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.regularization_scale)

        activation_function, initializer_function = None, None
        if activation == "tanh":
            activation_function = tf.nn.tanh
        elif activation == "relu":
            activation_function = tf.nn.relu

        if initializer == "he_init":
            initializer_function = tf.contrib.layers.variance_scaling_initializer()
        elif initializer == "xavier_init":
            initializer_function = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope(name):
            layer1 = tf.layers.dense(inputs=tensor, units=64, activation=activation_function,
                                     kernel_initializer=initializer_function, kernel_regularizer=l2_regularizer)
            dropout_layer1 = tf.layers.dropout(layer1, rate=self.dropout_rate, training=True)
            layer2 = tf.layers.dense(inputs=dropout_layer1, units=128, activation=activation_function,
                                     kernel_initializer=initializer_function, kernel_regularizer=l2_regularizer)
            dropout_layer2 = tf.layers.dropout(layer2, rate=self.dropout_rate, training=True)
            layer3 = tf.layers.dense(inputs=dropout_layer2, units=128, activation=activation_function,
                                     kernel_initializer=initializer_function, kernel_regularizer=l2_regularizer)
            dropout_layer3 = tf.layers.dropout(layer3, rate=self.dropout_rate, training=True)
            output_layer = tf.layers.dense(inputs=dropout_layer3, units=self.action_no,
                                           kernel_initializer=initializer_function)
        return output_layer

    def adjust_epsilon(self, step):
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * step)

    def epsilon_greedy_action(self, q_values, step):
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_no - 1)
        else:
            action = np.argmax(q_values)
        self.adjust_epsilon(step)
        return action

    def learn(self, episode_number=None):

        episode_num = self.episode_number if episode_number is None else episode_number
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            init.run()

            for episode in range(episode_num):
                state = self.env.reset()
                print("Starting state is " + str(state))
                episode_reward = 0
                iteration = 0
                done = False
                action = self.env.action_space.sample()

                while not done:
                    prev_state, prev_action = state, action
                    state, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                    #q_network_actions = self.q_network.eval(feed_dict={self.observation_format: np.expand_dims(state, 0)})

                    #print("Shape is " + str(state.shape))
                    q_network_actions = None
                    if state.shape == (3,):
                        q_network_actions = self.q_network.eval(feed_dict={self.observation_format: np.expand_dims(state, 1)})
                    elif state.shape == (3, 1):
                        q_network_actions = self.q_network.eval(feed_dict={self.observation_format: state})

                    next_action = self.epsilon_greedy_action(step=episode, q_values=q_network_actions)
                    action = next_action
                    self.memory.store(prev_state=prev_state, state=state, action=prev_action, reward=reward, done=done)

                    if iteration >= self.memory_required:
                        prev_state_batch, state_batch, action_batch, reward_batch, done_batch = self.memory.get_batch(
                            self.batch_size)
                        q_network_actions, target_network_actions = sess.run(
                            [self.q_network, self.target_network], feed_dic={self.observation_format: state_batch}
                        )
                        q_network_batch = reward_batch + self.gamma * np.amax(q_network_actions, axis=1) * \
                                          (1 - done_batch)
                        target_network_batch = reward_batch + self.gamma * np.amax(target_network_actions, axis=1) * \
                                          (1 - done_batch)
                        self.train_q_network.run(
                            feed_dic={
                                self.observation_format: prev_state_batch,
                                self.action_format: action_batch,
                                self.reward_format: target_network_batch
                            }
                        )
                        self.train_target_network.run(
                            feed_dic={
                                self.observation_format: prev_state_batch,
                                self.action_format: action_batch,
                                self.reward_format: q_network_batch
                            }
                        )

                    iteration += 1

                self.all_rewards.append(episode_reward)
        averaged_rewards = []
        av = 50
        for i in range(len(self.all_rewards) - av):
            averaged_reward = 0
            for j in range(av):
                averaged_reward += self.all_rewards[j+i]
            averaged_rewards.append(30 + averaged_reward/av)

        plt.plot(averaged_rewards)
        plt.ylabel("Reward")
        plt.xlabel("Episode Number")
        plt.show()

        return averaged_rewards

    def predict(self):
        pass


class ExperienceReplay(object):

    def __init__(self, max_length=10000, seed=None):
        self.memory_buffer = deque(maxlen=max_length)
        self.seed = random.seed(seed)

    def store(self, prev_state, state, action, reward, done):
        self.memory_buffer.append([prev_state, state, action, reward, done])

    def get_batch(self, batch_size):
        if self.get_buffer_size() < batch_size:
            batch = random.sample(self.memory_buffer, len(self.memory_buffer))
        else:
            batch = random.sample(self.memory_buffer, batch_size)

        # Do we need to put np.array([]) square brackets???
        prev_state = np.array(x[0] for x in batch)
        state = np.array(x[1] for x in batch)
        action = np.array(x[2] for x in batch)
        reward = np.array(x[3] for x in batch)
        done = np.array(x[4] for x in batch)

        return prev_state, state, action, reward, done

    def get_buffer_size(self):
        return len(self.memory_buffer)


myagent = DDQN()
