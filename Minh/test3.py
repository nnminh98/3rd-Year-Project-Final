import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from SinglePacketRoutingEnv import SinglePacketRoutingEnv

nodes1 = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"]
links1 = [["n1", "n2", 1], ["n1", "n0", 1], ["n0", "n2", 1], ["n3", "n0", 1], ["n2", "n5", 2], ["n3", "n5", 1],
              ["n4", "n3", 1], ["n4", "n6", 10], ["n6", "n3", 1], ["n4", "n7", 1], ["n6", "n7", 1], ["n5", "n6", 1]]

#env = SinglePacketRoutingEnv(nodes1, links1)
env = gym.make('CartPole-v0')

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("deepq_cartpole")

#del model # remove to demonstrate saving and loading

#model = DQN.load("deepq_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()