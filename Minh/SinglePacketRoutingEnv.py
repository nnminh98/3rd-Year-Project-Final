from gym import error, spaces, utils
import numpy as np
from Architecture import Network, Node, Link
from RoutingControllers import RoutingAlgorithm, RandomRouting, Dijkstra
from SimComponents import Packet
import random
from BaseEnvironment import BaseEnv


class SinglePacketRoutingEnv(BaseEnv):

    def __init__(self, nodes, edges, seed=None, packet=None):
        self.__version__ = "1.0.0"
        self.name = "Single Packet Routing Environment"
        super().__init__()
        self.seed = seed

        self.graph = self.create_network(nodes=nodes, edges=edges)

        self.finished = False
        self.step_number = -1
        self.episode_number = 0
        self.num_nodes = len(self.graph.nodes.values())

        # State = [current node, source node, next node]
        self.state = self.initial_state(packet=packet, seed=self.seed)
        self.past_state = None
        [self.state_np] = self.convert_state([self.state])
        [self.current_node] = self.get_current_nodes_from_state([self.state])
        [self.end_node] = self.get_end_nodes_from_state([self.state])

        self.max_action = self.get_max_action_integer()
        self.action_space = spaces.Discrete(self.max_action)
        high = np.array([self.num_nodes - 1, self.num_nodes - 1, self.num_nodes - 1])
        low = np.array([0, 0, 0])
        self.observation_space = spaces.Box(low=0, high=self.num_nodes-1, dtype=np.int, shape=(3,))

        self.reward = None
        #self.packet = None

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
        #self.packet = pkt
        self.graph.add_packet(pkt=pkt)

        state = [
            self.graph.nodes[src],
            self.graph.nodes[src],
            self.graph.nodes[dst],
        ]

        #print([state[0].id, state[1].id, state[2].id])
        return state

    def reset(self):
        self.graph.clear_packets()
        #del self.packet
        self.episode_number += 1
        self.step_number = -1
        self.finished = False
        self.state = self.initial_state(seed=self.seed)
        self.state_np = self.convert_state([self.state])
        [self.current_node] = self.get_current_nodes_from_state([self.state])
        [self.end_node] = self.get_end_nodes_from_state([self.state])

        return self.state_np

    def reset_episode_count(self):
        self.episode_number = 0

    def step(self, action):
        self.step_number += 1
        print(" ")
        print("Step" + str(self.step_number))
        try:
            #print("Action here is " + str(action))
            #print("Current node is " + str(self.current_node.id))
            selected_action, selected_link = self.current_node.routing_algorithm.set(action=action)

            self.env.run(until=self.env.now+1)

            self.past_state = self.state
            [self.state] = self.get_state()
            [self.state_np] = self.convert_state([self.state])
            [self.current_node] = self.get_current_nodes_from_state([self.state])
            self.finished = self.is_finished([self.state])
            # self.reward = self.get_reward(action=selected_action, state=self.past_state, link=selected_link)
            self.reward = self.get_reward(self.finished)

        except IndexError as e:
            print("index error")
            self.reward = -10

        if self.graph.packets[0].ttl <= self.graph.packets[0].ttl_safety or 100 < self.step_number:
            self.finished = True
            self.reward = -100

        if self.finished:
            self.reward = self.get_reward(self.finished)

        return self.state_np, self.reward, self.finished, {}

    @staticmethod
    def get_reward2(action, state, link):
        reward = None
        if action.id == state[2].id and action is not None:
            reward = 1
        elif link is not None:
            reward = -float(link.cost)/100
        return reward

    def get_reward(self, done):
        if not done:
            return 0
        else:
            try:
                return 10 / self.graph.packets[0].total_route_weight
            except ZeroDivisionError as e:
                print("The path for packet [{} {}] is ".format(self.graph.packets[0].src, self.graph.packets[0].dst) + str(self.graph.packets[0].path))
