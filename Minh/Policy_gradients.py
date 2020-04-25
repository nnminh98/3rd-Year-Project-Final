import os
import logging
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#import gym_network
#from log_setup import init_logging
from SinglePacketRoutingEnv import SinglePacketRoutingEnv


def discount_rewards(rewards, gamma=0.95):
    rewards = [reward * np.power(gamma, index) for index, reward in enumerate(rewards)]
    print(type(rewards))
    for iteration in range(rewards.__len__()):
        rewards[iteration] = sum(rewards[iteration:])
    rewards = np.divide(
        rewards - np.mean(rewards), np.std(rewards) + np.finfo(np.float32).eps
    )
    print(type(rewards))
    return rewards


def finalize_episode(iteration, rewards, average_reward):
    print(f"Episode {iteration} and Episode Rewards: {sum(rewards)} and Average Reward: {average_reward}")
    #logging.info(f"Episode {iteration} and Episode Rewards: {sum(rewards)} and Average Reward: {average_reward}")


def buildDeepNetwork(nb_actions):
    weight_init = tf.initializers.random_normal(mean=0.0, stddev=0.1)
    input_ = tf.placeholder(dtype=np.float32, shape=(None, 3), name="input_")
    fc1 = tf.contrib.layers.fully_connected(
        inputs=input_,
        num_outputs=128,
        activation_fn=tf.nn.relu,
        weights_initializer=weight_init,
    )
    fc2 = tf.contrib.layers.fully_connected(
        inputs=fc1,
        num_outputs=128,
        activation_fn=tf.nn.relu,
        weights_initializer=weight_init,
    )
    fc3 = tf.contrib.layers.fully_connected(
        inputs=fc2,
        num_outputs=64,
        activation_fn=tf.nn.relu,
        weights_initializer=weight_init,
    )
    fc4 = tf.contrib.layers.fully_connected(
        inputs=fc3,
        num_outputs=nb_actions,
        activation_fn=None,
        weights_initializer=weight_init,
    )
    return fc4


nodes1 = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"]
links1 = [["n1", "n2", 1], ["n1", "n0", 1], ["n0", "n2", 1], ["n3", "n0", 1], ["n2", "n5", 2], ["n3", "n5", 1],
              ["n4", "n3", 1], ["n4", "n6", 10], ["n6", "n3", 1], ["n4", "n7", 1], ["n6", "n7", 1], ["n5", "n6", 1]]

env_new = SinglePacketRoutingEnv(nodes=nodes1, edges=links1)


def runPolicyGradients(env='PathFindingNetworkEnv-v1', network="germany50", render=False, mode="human", log_level="DEBUG", seed=0, num_episodes=2500):

    """
    Runs Policy Gradients Deep RL on Germany50 network:
    network: String, filename without the .xml extension of the network to be used
             e.g. 'germany50', 'newyork'
    render: Boolean, whether to render the network animation or not, if True considerably slows down training
    mode: either 'human' or 'rgb_array', how to render the animation
    logging: either 'CRITICAL', 'ERROR', 'WARNING', 'INFO' or 'DEBUG',
             logging level to use for the current run, 'CRITICAL' shows the least info, 'DEBUG' shows the most.
    seed: Integer or None, used to set the seed for selecting the start and end nodes, defaults to 0. Keeping this a constant value will cause the same start and end nodes to be chosen each function call."
    """
    #init_logging(max_log_files=10, logging_level=log_level)

    #logging.info("Running Policy Gradients for {} episodes.".format(str(num_episodes)))

    ENV_NAME = env
    kwargs = {"network": network, "seed": None}
    #env = gym.make(ENV_NAME, **kwargs)
    #env = gym.make('CartPole-v0')
    env = SinglePacketRoutingEnv(nodes=nodes1, edges=links1)

    nb_actions = env.action_space.n
    # print("NB_ACTIONS: ", nb_actions)

    #used for graphing:
    reward_history = []
    reward_step = 10

    episode_actions_ = tf.placeholder(
        dtype=np.float32, shape=(None, nb_actions), name="episode_actions_"
    )
    dicounted_episode_rewards_ = tf.placeholder(
        dtype=np.float32, shape=(None,), name="discounted_episode_rewards_"
    )

    fc4 = buildDeepNetwork(nb_actions)

    action_distribution = tf.nn.softmax(fc4)

    neg_loss_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=fc4, labels=episode_actions_
    )
    loss = tf.reduce_mean(neg_loss_prob * dicounted_episode_rewards_)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    total_reward_for_run = 0

    with tf.Session() as sess:
        episode_count = 0
        sess.run(tf.global_variables_initializer())
        episode_states, episode_actions, episode_rewards = [], [], []
        for episode in range(num_episodes + 1):
            episode_count += 1
            state = env.reset()
            while True:
                probability_distribution = sess.run(
                    action_distribution,
                    feed_dict={"input_:0": state.reshape([1, 3])},
                )
                action = np.random.choice(
                    range(nb_actions), p=probability_distribution.ravel()
                )
                new_state, reward, done, _ = env.step(action)

                # if render parameter passed as True, will display a render of the network,
                # the current start and end nodes, and the nodes visited during the iteration
                if render:
                    env.render(mode=mode)

                episode_states.append(state)
                action_ = [0 if index != action else 1 for index in range(nb_actions)]
                episode_actions.append(action_)
                episode_rewards.append(reward)
                total_reward_for_run += sum(episode_rewards)
                average_reward = total_reward_for_run/(episode+1)
                # print(reward)
                if done:
                    episode_reward = sum(episode_rewards)
                    reward_history.append(episode_reward)
                    discounted_rewards = discount_rewards(episode_rewards)
                    sess.run(
                        optimizer,
                        feed_dict={
                            "input_:0": episode_states,
                            "episode_actions_:0": episode_actions,
                            "discounted_episode_rewards_:0": discounted_rewards,
                        },
                    )
                    finalize_episode(episode, episode_rewards, average_reward)
                    episode_states, episode_actions, episode_rewards = [], [], []
                    break
                state = new_state
        #print(reward_history)
        rewards = [np.average(reward_history[index*reward_step:(index+1)*reward_step]) for index in range(int(episode_count/reward_step))]
        print(rewards)
        #plt.plot(reward_history)
        #plt.show()
        new_averaged_rewards = []
        average_batch_size = 25
        for i in range(len(reward_history) - average_batch_size):
            averaged_reward = 0
            for j in range(average_batch_size):
                averaged_reward += reward_history[i + j]
            new_averaged_rewards.append(averaged_reward / average_batch_size)
        return new_averaged_rewards


#runPolicyGradients()

