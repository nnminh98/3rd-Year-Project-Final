from SinglePacketRoutingEnv import SinglePacketRoutingEnv
from LinkFailureEnv import LinkFailureEnv
from DijkstraAgent import DijkstraAgent
import matplotlib.pyplot as plt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import ACKTR, DQN


def create_node_param(node_num):
    nodes = []
    for i in range(node_num):
        nodes.append("n{}".format(i))
    return nodes


nodes1 = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"]
links1 = [["n1", "n2", 1], ["n1", "n0", 1], ["n0", "n2", 1], ["n3", "n0", 1], ["n2", "n5", 2], ["n3", "n5", 1], ["n4", "n3", 1], ["n4", "n6", 10], ["n6", "n3", 1], ["n4", "n7", 1],  ["n6", "n7", 1],  ["n5", "n6", 1]]

nodes2 = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12", "n13", "n14", "n15", "n16",
          "n17", "n18", "n19", "n20", "n21", "n22", "n23", "n24", "n25"]
links2 = [["n15", "n16", 1], ["n15", "n17", 1], ["n15", "n13", 10], ["n14", "n13", 1], ["n15", "n18", 10],
          ["n13", "n7", 10], ["n10", "n7", 1], ["n9", "n7", 1], ["n11", "n7", 1], ["n8", "n7", 1], ["n7", "n12", 1],
          ["n7", "n18", 1], ["n18", "n19", 1], ["n7", "n4", 1], ["n4", "n5", 1], ["n4", "n6", 1], ["n4", "n3", 10],
          ["n1", "n3", 1], ["n4", "n20", 1], ["n20", "n21", 1], ["n20", "n22", 1], ["n20", "n23", 1],
          ["n20", "n24", 1], ["n20", "n25", 1], ["n2", "n3", 10], ["n20", "n2", 10], ["n0", "n2", 1], ["n18", "n20", 1]]

myEnv = SinglePacketRoutingEnv(nodes=nodes2, edges=links2, seed=None, packet=None)
state = [myEnv.state[0].id, myEnv.state[1].id, myEnv.state[2].id]
print(state)
# myEnv.step(0)
"""for i in range(20):
    myEnv.step(0)
state = [myEnv.state[0].id, myEnv.state[1].id, myEnv.state[2].id]
print(state)"""

"""myEnv.reset()
for _ in range(100):
    myEnv.render()
    state, reward, done, info = myEnv.step(myEnv.action_space.sample())
    print(reward)

print(myEnv.observation_space.sample())"""

myAgent = DijkstraAgent(env=myEnv, nodes=nodes2, edges=links2)
dijkstra_rewards = myAgent.return_results()

env = SinglePacketRoutingEnv(nodes=nodes2, edges=links2)
# Stable Baselines provides you with make_vec_env() helper
# which does exactly the previous steps for you:
# env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

print("Learning")
model = ACKTR(MlpPolicy, env, verbose=1)
#model = DQN(LnMlpPolicy, env, verbose=1)
model.learn(total_timesteps=5000)
ACKTR_results = model.return_results()

avareged_ACKTR = []
av = 70
for i in range(len(ACKTR_results) - av):
    averaged_reward = 0
    for j in range(av):
        averaged_reward += ACKTR_results[j+i]
    avareged_ACKTR.append(averaged_reward/av)

plt.plot(avareged_ACKTR)
plt.plot(dijkstra_rewards)
plt.ylabel("Reward")
plt.xlabel("Episode Number")
plt.show()


from gym import error, spaces, utils
import numpy as np
from Architecture import Network, Node, Link
from RoutingControllers import RoutingAlgorithm, RandomRouting, Dijkstra
from SimComponents import Packet
import random
from random import gauss
import math
from BaseEnvironment import BaseEnv


class TrafficEnv(BaseEnv):

    def __init__(self, nodes, edges, seed=None, packet=None):
        self.__version__ = "1.0.0"
        self.name = "Traffic Routing Env"
        super().__init__()
        self.seed = seed

        self.graph = self.create_network(nodes=nodes, edges=edges)

        self.finished = False
        self.step_number = -1
        self.episode_number = 0
        self.num_nodes = len(self.graph.nodes.values())

        self.max_action = self.get_max_action_integer()
        self.action_space = spaces.Discrete(self.max_action)
        self.observation_space = spaces.Box(low=0, high=self.num_nodes - 1, dtype=np.int, shape=(3,))

        self.state = self.initial_state(packet=packet, seed=self.seed)
        self.past_state = None
        [self.state_np] = self.convert_state([self.state])
        [self.current_node] = self.get_current_nodes_from_state([self.state])
        [self.end_node] = self.get_end_nodes_from_state([self.state])

        self.reward = None

        self.links_info = self.graph.links.copy()
        self.clear_traffic()


    def initial_state(self, seed=None, packet=None):
        if seed is None:
            random.seed()
        else:
            random.seed(seed)

        if packet is None:
            src = random.choice(list(self.graph.nodes.keys()))
            dst = random.choice(list(self.graph.nodes.keys()))
            while dst == src:
                dst = random.choice(list(self.graph.nodes.keys()))

        else:
            src = packet[0]
            dst = packet[1]

        pkt = Packet(time=self.graph.env.now, size=1, id=random.randint(1, 100), src=src, dst=dst)
        self.graph.add_packet(pkt=pkt)

        state = [
            self.graph.nodes[src],
            self.graph.nodes[src],
            self.graph.nodes[dst],
        ]

        # print([state[0].id, state[1].id, state[2].id])
        return state

    def step(self, action):
        self.step_number += 1
        print(" ")
        print("Step" + str(self.step_number))
        try:
            selected_action, selected_link = self.current_node.routing_algorithm.set(action=action)
            self.links_info[selected_link.id] += 1

            self.env.run(until=self.env.now + 1)

            self.past_state = self.state
            self.state = self.get_state()
            [self.state_np] = self.convert_state([self.state])
            [self.current_node] = self.get_current_nodes_from_state([self.state])
            self.finished = self.is_finished([self.state])
            # self.reward = self.get_reward(action=selected_action, state=self.past_state, link=selected_link)
            self.reward = self.get_reward(self.finished)

        except IndexError as e:
            print("index error")
            self.reward = -10

        if self.graph.packets[0].ttl <= self.graph.packets[0].ttl_safety or 50 < self.step_number:
            self.finished = True
            self.reward = -1000

        if self.finished:
            self.reward = self.get_reward(done=self.finished)

        return self.state_np, self.reward, self.finished, {}

    def reset(self):
        self.clear_traffic()
        self.adjust_traffic()

        self.step_number = -1
        self.graph.clear_packets()
        self.episode_number += 1
        self.finished = False
        self.state_np = self.convert_state([self.state])
        [self.current_node] = self.get_current_nodes_from_state([self.state])
        [self.end_node] = self.get_end_nodes_from_state([self.state])

        return self.state_np

    def get_reward(self, done):
        if not done:
            return 0
        else:
            if self.graph.packets[0].delivered:
                return 10 / self.graph.packets[0].total_traffic
            else:
                return -1000

    def adjust_traffic(self):
        for link in self.links_info.keys():
            variance = 7
            mean = 50 + self.links_info[link] * 2
            self.graph.links[link].traffic = gauss(mean, math.sqrt(variance))

    def clear_traffic(self):
        for link in self.links_info.keys():
            self.links_info[link] = 0

        for link in self.graph.links.values():
            link.traffic = 50
