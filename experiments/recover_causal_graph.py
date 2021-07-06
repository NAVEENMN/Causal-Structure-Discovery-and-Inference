import os
import sys
import random
import pandas as pd
import seaborn as sns
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg


def nPr(_set, r):
    from itertools import permutations
    return list(permutations(_set, r))

def draw_graph(g, ax=None):
    edges = g.edges()
    colors = nx.get_edge_attributes(g, 'color').values()
    weights = nx.get_edge_attributes(g, 'weight').values()
    pos = nx.circular_layout(g)
    nx.draw(g, pos,
            with_labels=True,
            width=list(weights),
            node_size=500,
            ax=ax)


class Node(object):
    def __init__(self, name, value=0.0):
        self.name = name
        self.mu = np.random.normal(0, 1, 1)
        self.sd = 1
        self._value = np.asarray(value) + np.random.normal(0.0, 1.0, 1) / 100000.0

    def __repr__(self):
        return f'{self.name}: {self._value}'

    def reset(self):
        _noise = np.random.normal(0.0, 1.0, 1) / 100000.0
        self._value = 0.0 + _noise

    def get_mu_sd(self):
        return self.mu, self.sd

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value

    def update_value(self, value):
        self._value += np.asarray(value)

class CausalGraph(object):
    def __init__(self):
        self.mediators_count = 0
        self.forks = 0
        self.colliders = 0
        self.g = nx.DiGraph()
        self.nodes = {}

    def __repr__(self):
        return self.g

    def create_a_node(self):
        _node = f'n_{self.g.number_of_nodes()}'
        self.g.add_node(_node)
        self.nodes[_node] = Node(name=_node)
        return _node

    def reset(self):
        for node in self.nodes.keys():
            self.nodes[node].reset()

    def get_total_nodes(self):
        return self.g.number_of_nodes()

    def get_node_names(self):
        return self.g.nodes()

    # x -> y -> z
    def _add_a_mediator(self):

        color_code = 'b'

        node = None
        if self.g.number_of_nodes() != 0:
            node = random.sample(self.g.nodes(), 1)[0]

        x = self.create_a_node()
        y = self.create_a_node()
        z = self.create_a_node()

        if random.randint(0, 1):
            self.g.add_edge(x, y, color=color_code, weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))
            self.g.add_edge(y, z, color=color_code, weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))
        else:
            self.g.add_edge(z, y, color=color_code, weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))
            self.g.add_edge(y, x, color=color_code, weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))

        if node:
            link = random.sample([x, y, z], 1)[0]
            if random.randint(0, 1):
                self.g.add_edge(node, link, color=color_code, weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))
            else:
                self.g.add_edge(link, node, color=color_code, weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))

    # x <- y -> z
    def _add_a_fork(self):
        color_code = 'b'

        node = None
        if self.g.number_of_nodes() != 0:
            node = random.sample(self.g.nodes(), 1)[0]

        x = self.create_a_node()
        y = self.create_a_node()
        z = self.create_a_node()

        self.g.add_edge(y, x, color=color_code, weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))
        self.g.add_edge(y, z, color=color_code, weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))
        if node:
            link = random.sample([x, y, z], 1)[0]
            if random.randint(0, 1):
                self.g.add_edge(node, link, color=color_code, weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))
            else:
                self.g.add_edge(link, node, color=color_code, weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))

    # x -> y <- z
    def _add_a_collider(self):
        color_code = 'b'

        node = None
        if self.g.number_of_nodes() != 0:
            node = random.sample(self.g.nodes(), 1)[0]

        x = self.create_a_node()
        y = self.create_a_node()
        z = self.create_a_node()

        self.g.add_edge(x, y, color=color_code, weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))
        self.g.add_edge(z, y, color=color_code, weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))
        if node:
            link = random.sample([x, y, z], 1)[0]
            if random.randint(0, 1):
                self.g.add_edge(node, link, color=color_code, weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))
            else:
                self.g.add_edge(link, node, color=color_code, weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))

    def generate_random_graph(self, mediators=1, forks=1, colliders=1):
        for _ in range(mediators):
            self._add_a_mediator()
        for _ in range(forks):
            self._add_a_fork()
        for _ in range(colliders):
            self._add_a_collider()

    def get_edges(self):
        return self.g.edges()

    def get_graph(self):
        return self.g

    def show_graph(self, ax=None):
        edges = self.g.edges()
        colors = nx.get_edge_attributes(self.g, 'color').values()
        weights = nx.get_edge_attributes(self.g, 'weight').values()
        pos = nx.circular_layout(self.g)
        nx.draw(self.g, pos,
                with_labels=True,
                edge_color=list(colors),
                width=list(weights),
                node_size=500,
                ax=ax)
        plt.show()

    def _update_graph(self, _nodes):
        _edges = nx.get_edge_attributes(self.g, 'weight')
        for _edge in _edges:
            # a -> b
            (a, b) = _edge
            _nodes[b] += (_edges[_edge] * _nodes[a])
        return _nodes

    def generate_observation(self, independent_variables):

        # initialize values to independent_variables
        for variable in independent_variables:
            nd = self.nodes[variable]
            sample_var = lambda _mu, _sd: np.random.normal(_mu, _sd, 1)
            m, s = 0.5, 1
            nd.set_value(sample_var(m, s))

        def breadth_first_traversal(q, g):
            while q:
                node = q.pop(0)
                v = self.nodes.get(node).get_value()
                q.extend(list(g.successors(node)))
                for successor in list(g.successors(node)):
                    edge_value = self.g.get_edge_data(node, successor)
                    w, c = edge_value['weight'], edge_value['capacity']
                    self.nodes.get(successor).update_value((v*w)+c)
            return

        q = []
        for variable in independent_variables:
            q.append(variable)
            breadth_first_traversal(q, self.g)

        return self.nodes.values()

    def get_observations(self, n=0):
        observations = {}
        self.reset()

        # Get all independent variables
        independent_variables = []
        for node in self.g.nodes():
            if self.g.in_degree(node) == 0:
                independent_variables.append(node)

        for step in range(n):
            self.generate_observation(independent_variables)
            for node in self.nodes:
                if node in observations.keys():
                    observations[node].append(self.nodes[node].get_value()[0])
                else:
                    observations[node] = [self.nodes[node].get_value()[0]]
            self.reset()
        return pd.DataFrame(observations)


def compute_correlation_between_nodes(cg, observations):
    observations = observations.T
    corr = np.corrcoef(x=observations)
    _index = dict()
    _index = {i: f'n_{i}' for i in range(0, cg.get_total_nodes())}
    df_corr = pd.DataFrame(data=corr, columns=cg.get_node_names())
    df_corr.index = cg.get_node_names()
    return df_corr


def breadth_first_traversal(q, g):
    while q:
        print(q)
        node = q.pop(0)
        print(f'visiting {node}')
        print(f'adding childrens {list(g.successors(node))}')
        q.extend(list(g.successors(node)))
    return


def main():
    cg = CausalGraph()
    cg.generate_random_graph(mediators=3, forks=0, colliders=0)
    gp = cg.get_graph()
    # Generate few observations
    observations = cg.get_observations(n=500)
    print(observations.head())

    # case a: empty z
    _observations = observations.T
    pearson_correlation = np.corrcoef(x=_observations)
    #print(pearson_correlation)

    complete_graph = nx.complete_graph(cg.get_node_names())

    # case a: empty z
    nodes = nPr(cg.get_node_names(), 2)
    for (x, y) in nodes:
        corr = np.corrcoef(observations[x], observations[y])
        corr = corr[0][1]
        #print(x, y)
        #print(f'{x, y} - {corr}')
        if complete_graph.has_edge(x, y) and (np.abs(corr) < 0.9):
            complete_graph.remove_edge(x, y)

    # case a: empty z
    nodes = nPr(cg.get_node_names(), 3)
    for (x, y, z) in nodes:
        # print(f'Partial correlation between {(x, y)} and {z}')
        df = pd.DataFrame({'x': observations[x], 'y': observations[y], 'z': observations[z]})
        result = pg.partial_corr(data=df, x='x', y='y', covar='z').round(3)
        p_value = result['p-val']['pearson']
        r_value = result['r']['pearson']
        if p_value > 0.08:
            #print(f'{x, y} - {p_value} - {r_value}')
            if complete_graph.has_edge(x, y):
                complete_graph.remove_edge(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('Original Graph')
    axes[1].set_title('Predicted Graph')



    draw_graph(g=cg.get_graph(), ax=axes[0])
    draw_graph(g=complete_graph, ax=axes[1])
    #plt.show()

    #sns.scatterplot(x=observations.n_3, y=observations.n_5)
    #sns.scatterplot(x=observations.n_0, y=observations.n_3)
    #sns.scatterplot(x=observations.n_0, y=observations.n_1)
    sns.pairplot(observations)
    plt.show()




if __name__ == "__main__":
    main()
