import random
import argparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=10,
                    help='Number of nodes in the graph.')
parser.add_argument('--conn', type=float, default=0.4,
                    help='degree of connectedness')
parser.add_argument('--samples', type=int, default=100,
                    help='number of samples')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class Graph:
    def __init__(self, num_of_nodes):
        self.gh = nx.DiGraph()
        self.num_of_nodes = num_of_nodes
        self.node_names = [f"node_{i}" for i in range(num_of_nodes)]
        self.means = np.random.random_sample((num_of_nodes,))
        self.sd = np.random.random_sample((num_of_nodes,))

    def get_nodes(self):
        return self.gh.nodes()

    def get_edges(self):
        return self.gh.edges()

    def set_number_of_nodes(self, n):
        self.num_of_nodes = n

    def update_nodes(self):
        # V(node_b) = V(node_b) + Weight * V(node_a)
        for edge in self.gh.edges:
            node_a_value = self.gh.nodes.get(edge[0]).get("value")
            edge_weight = self.gh.get_edge_data(edge[0], edge[1])['weight']
            node_b_value = self.gh.nodes.get(edge[1]).get("value")
            result = node_a_value + (edge_weight * node_b_value)
            self.gh.nodes[edge[1]]['value'] = result

    def print(self):
        logging.debug("Printing Graph")
        nx.draw(self.gh, pos=nx.circular_layout(self.gh), node_color='r', edge_color='b', with_labels=True)
        plt.savefig('../../data/graph.png')
        logging.debug("Graph saved at ../../data/graph.png")
        plt.show()

    def get_readings(self):
        data = {node: [self.gh.nodes.get(node).get("value")] for node in self.gh.nodes()}
        data['timestamp'] = datetime.datetime.now()
        return pd.DataFrame(data)

    def reinit(self):
        logging.debug("Reinitializing node values")
        for idx, node in enumerate(self.node_names):
            _node = self.gh.nodes.get(node)
            _node.update({'value': np.random.normal(self.means[idx], self.sd[idx])})

    def generate_random_graph(self):
        logging.debug("Generating random graph")
        # Add nodes
        for idx, node in enumerate(self.node_names):
            self.gh.add_node(node, value=np.random.normal(self.means[idx], self.sd[idx]))

        # Add edges
        for i in range(self.num_of_nodes):
            for j in range(i + 1, self.num_of_nodes):
                # Add an edge with probability conn
                if random.random() < args.conn:
                    self.gh.add_edge(f"node_{i}", f"node_{j}", weight=1.0)
