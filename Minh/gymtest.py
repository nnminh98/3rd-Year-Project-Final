import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import ACKTR
from SinglePacketRoutingEnv import SinglePacketRoutingEnv
from DijkstraAgent import DijkstraAgent
#from DQN import DDQN


env = gym.make('CartPole-v1')
model = ACKTR(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=2500)

obs = env.reset()
while True:
    action, states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()


