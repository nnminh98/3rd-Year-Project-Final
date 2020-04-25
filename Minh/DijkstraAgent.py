import gym
import sys
import numpy as np
import matplotlib.pyplot as plt


class DijkstraAgent(object):

    def __init__(self, env, nodes, edges, openAI_env="Single Packet Routing Environment", packet=None):
        self.env_name = openAI_env
        self.kwargs = {"nodes": nodes, "edges": edges}
        self.env = env# gym.make(self.env_name, **self.kwargs)
        self.nodes = self.env.graph.nodes
        self.rewards = []

        for i in range(3500):
            if i is not 0:
                self.env.reset()
            print("")
            print("-----------------------------------------------")
            print("EPISODE " + str(self.env.episode_number))
            self.initial_state = self.env.state
            print(self.env.convert_state([self.initial_state]))
            self.src, self.dst = self.initial_state[1], self.initial_state[2]
            reward = self.__call__()
            self.rewards.append(reward)

        # print(np.average(self.rewards))
        print(self.rewards)
        print(np.average(self.rewards))
        #plt.plot(self.rewards)
        #plt.show()

        self.new_averaged_rewards = []
        average_batch_size = 10
        for i in range(len(self.rewards) - average_batch_size):
            averaged_reward = 0
            for j in range(average_batch_size):
                averaged_reward += self.rewards[i + j]
            self.new_averaged_rewards.append(3 * averaged_reward / average_batch_size - 350)

    def rewards(self):
        return self.rewards

    def __call__(self):

        def min_distance(distance, spt_set, self_nodes):
            """Returns the node with the minimum distance value that has not yet been added
            :param distance:
            :param spt_set:
            :param self_nodes:
            :return: Node object
            """
            minimum = sys.maxsize
            minimum_node = None
            for curr_node in self_nodes.values():
                if distance[curr_node.id] < minimum and not spt_set[curr_node.id]:
                    minimum = distance[curr_node.id]
                    minimum_node = curr_node
            return minimum_node

        done = False
        current_state = self.initial_state
        episode_reward = 0

        while not done:
            src = current_state[0]
            dst = current_state[2]

            distances = self.env.graph.nodes.copy()
            for node in distances.keys():
                distances[node] = sys.maxsize
            distances[src.id] = 0

            spt_set = self.env.graph.nodes.copy()
            for node in spt_set.keys():
                spt_set[node] = False

            path = self.env.graph.nodes.copy()
            for node in path.keys():
                path[node] = []
            #path[src.id]

            for count in range(len(self.env.graph.nodes)):
                current = min_distance(distance=distances, spt_set=spt_set, self_nodes=self.env.graph.nodes)
                spt_set[current.id] = True
                if current == dst:
                    break

                for v in self.env.graph.nodes.values():
                    if current.is_neighbour(v) and not spt_set[v.id] and distances[v.id] > distances[current.id] + current.routes["{}_{}".format(current.id, v.id)].cost:
                        if current.routes["{}_{}".format(current.id, v.id)].state:
                            distances[v.id] = distances[current.id] + current.routes["{}_{}".format(current.id, v.id)].cost
                            path[v.id] = path[current.id].copy()
                            path[v.id].append(v)

            next_node = path[dst.id][0]
            #print("Next node is " + str(next_node.id))
            src_neighbours = src.get_neighbour_id()
            #print("Neighbours are " + str(src_neighbours))
            action = None
            for i in range(len(src_neighbours)):
                if next_node.id == src_neighbours[i]:
                    action = i

            #print("Action is " + str(action))

            np_state, reward, done, dummy = self.env.step(action=action)
            [current_state] = self.env.convert_state_back([np_state])
            episode_reward += reward

        return episode_reward

    def return_results(self):
        return self.rewards