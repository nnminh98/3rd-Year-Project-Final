from gym import error, spaces, utils
import numpy as np
from Architecture import Network, Node, Link
from RoutingControllers import RoutingAlgorithm, RandomRouting, Dijkstra
from SimComponents import Packet
import random
from BaseEnvironment import BaseEnv


class LinkFailureEnv(BaseEnv):

    def __init__(self, nodes, edges, broken_link_num, seed=None, packet=None):
        self.__version__ = "1.0.0"
        self.name = "Link Failure Environment"
        super().__init__()
        self.seed = seed

        self.graph = self.create_network(nodes=nodes, edges=edges)

        self.finished = False
        self.step_number = -1
        self.episode_number = 0
        self.num_nodes = len(self.graph.nodes.values())

        self.max_action = self.get_max_action_integer()
        self.action_space = spaces.Discrete(self.max_action)
        self.observation_space = spaces.Box(low=0, high=self.num_nodes-1, dtype=np.int, shape=(3,))

        self.broken_link_num = broken_link_num
        self.broken_links = self.break_links(broken_link_num=self.broken_link_num)

        self.state = self.initial_state(packet=packet, broken_links=self.broken_links, seed=self.seed)
        self.past_state = None
        [self.state_np] = self.convert_state([self.state])
        [self.current_node] = self.get_current_nodes_from_state([self.state])
        [self.end_node] = self.get_end_nodes_from_state([self.state])

        self.reward = None

    def initial_state(self, broken_links, seed=None, packet=None):
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

        #print([state[0].id, state[1].id, state[2].id])
        return state

    def step(self, action):
        self.step_number += 1
        print(" ")
        print("Step" + str(self.step_number))
        try:
            selected_action, selected_link = self.current_node.routing_algorithm.set(action=action)

            self.env.run(until=self.env.now + 1)

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

        if self.graph.packets[0].ttl <= self.graph.packets[0].ttl_safety or 50 < self.step_number:
            self.finished = True
            self.reward = -1000

        if self.finished:
            self.reward = self.get_reward(done=self.finished)

        return self.state_np, self.reward, self.finished, {}

    def reset(self):
        for link in self.graph.links.keys():
            self.graph.links[link].__setstate__(state=True)
        self.broken_links = self.break_links(self.broken_link_num)

        self.step_number = -1
        self.graph.clear_packets()
        self.episode_number += 1
        self.finished = False
        self.state = self.initial_state(broken_links=self.broken_links)
        self.state_np = self.convert_state([self.state])
        [self.current_node] = self.get_current_nodes_from_state([self.state])
        [self.end_node] = self.get_end_nodes_from_state([self.state])

        return self.state_np

    def get_reward(self, done):
        if not done:
            return 0
        else:
            if self.graph.packets[0].delivered:
                return 10 / self.graph.packets[0].total_route_weight
            else:
                return -1000

    def break_links(self, broken_link_num, random_link_num=False):
        if random_link_num:
            broken_link_num = np.random.randint(0, len(self.graph.links))
        else:
            broken_link_num = broken_link_num

        links_broken = []

        for i in range(broken_link_num):
            link = random.choice(list(self.graph.links.keys()))
            link2 = self.graph.links[link].get_counter_id()

            while not self.graph.links[link].state:
                link = random.choice(list(self.graph.links.keys()))
                link2 = self.graph.links[link].get_counter_id()

            self.graph.links[link].__setstate__(state=False)
            self.graph.links[link2].__setstate__(state=False)

            links_broken.append(link)
            links_broken.append(link2)

        #print("Broken link is " + links_broken[0])
        return links_broken
