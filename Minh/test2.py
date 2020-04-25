import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import ACKTR, DQN
from SinglePacketRoutingEnv import SinglePacketRoutingEnv
from LinkFailureEnv import LinkFailureEnv
from DijkstraAgent import DijkstraAgent
from Policy_gradients import runPolicyGradients
from DQN import DDQN


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


if __name__ == '__main__':
    env_id = "CartPole-v1"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    nodes1 = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"]
    links1 = [["n1", "n2", 1], ["n1", "n0", 1], ["n0", "n2", 1], ["n3", "n0", 1], ["n2", "n5", 2], ["n3", "n5", 1],
              ["n4", "n3", 1], ["n4", "n6", 10], ["n6", "n3", 1], ["n4", "n7", 1], ["n6", "n7", 1], ["n5", "n6", 1]]

    nodes2 = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12", "n13", "n14", "n15",
              "n16", "n17", "n18", "n19", "n20", "n21", "n22", "n23", "n24", "n25"]
    links2 = [["n15", "n16", 1], ["n15", "n17", 1], ["n15", "n13", 10], ["n14", "n13", 1], ["n15", "n18", 10],
              ["n13", "n7", 10], ["n10", "n7", 1], ["n9", "n7", 1], ["n11", "n7", 1], ["n8", "n7", 1], ["n7", "n12", 1],
              ["n7", "n18", 1], ["n18", "n19", 1], ["n7", "n4", 1], ["n4", "n5", 1], ["n4", "n6", 1], ["n4", "n3", 10],
              ["n1", "n3", 1], ["n4", "n20", 1], ["n1", "n2", 1], ["n20", "n21", 1], ["n20", "n22", 1],
              ["n20", "n23", 1],
              ["n20", "n24", 1], ["n20", "n25", 1], ["n2", "n3", 10], ["n20", "n2", 10], ["n0", "n2", 1],
              ["n18", "n20", 1]]

    env = SinglePacketRoutingEnv(nodes=nodes1, edges=links1)
    env2 = LinkFailureEnv(nodes=nodes1, edges=links1, broken_link_num=0)
    env1 = gym.make('CartPole-v0')
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)
    model1 = ACKTR(MlpPolicy, env1, verbose=1, learning_rate=0.6)
    acktr_rewards1 = model1.learn(total_timesteps=2500)
    model2 = ACKTR(MlpPolicy, env1, verbose=1, learning_rate=0.1)
    acktr_rewards2 = model2.learn(total_timesteps=2500)
    model3 = ACKTR(MlpPolicy, env1, verbose=1)
    acktr_rewards3 = model3.learn(total_timesteps=2500)

    plt.plot(acktr_rewards1, label="learning_rate 0.4")
    plt.plot(acktr_rewards2, label="learning_rate 0.1")
    plt.plot(acktr_rewards3, label="learning_rate 0.25")
    plt.ylabel("Averaged Reward (in batches of 20)")
    plt.xlabel("Episode Number")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()