import random
import pandas as pd
import seaborn as sns
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg


class Node(object):
    def __init__(self, name, value=0.0):
        self.name = name
        self.mu = np.random.normal(0, 1, 1)
        self.sd = 1
        self._value = np.asarray(value) + np.random.normal(0.0, 1.0, 1) / 100000.0

    def __repr__(self):
        return f'{self.name}: {self._value}'

    def get_name(self):
        return self.name

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


class Graph(object):
    def __init__(self, gh):
        self.graph = gh
        self.node_size = 800
        self.node_color = '#0D0D0D'
        self.font_color = '#D9D9D9'
        self.edge_color = '#262626'

    def get_number_of_nodes(self):
        return self.graph.number_of_nodes()

    def get_random_node(self):
        node = None
        if self.graph.number_of_nodes() != 0:
            node = random.sample(self.graph.nodes(), 1)[0]
        return node

    def get_source_nodes(self):
        _nodes = []
        for node in self.graph.nodes():
            if self.graph.in_degree(node) == 0:
                _nodes.append(node)
        return _nodes

    def get_edge_value(self, node_a, node_b):
        return self.graph.get_edge_data(node_a, node_b)

    def get_graph(self):
        return self.graph

    def add_node_to_graph(self, node):
        self.graph.add_node(node, value=0.0)

    def add_an_edge_to_graph(self, node_a, node_b):
        self.graph.add_edge(node_a, node_b, color=self.edge_color,
                            weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))

    def draw_graph(self, axes):
        colors = nx.get_edge_attributes(self.graph, 'color').values()
        weights = nx.get_edge_attributes(self.graph, 'weight').values()
        nx.draw(self.graph,
                pos=nx.circular_layout(self.graph),
                with_labels=True,
                edge_color=list(colors),
                width=list(weights),
                node_size=500,
                ax=axes)


class CausalGraph(Graph):
    def __init__(self, gh):
        super().__init__(gh)
        self.left_mediators_count = 0
        self.right_mediators_count = 0
        self.forks_count = 0
        self.colliders_count = 0
        self.nodes = {}

    def __repr__(self):
        return self.get_graph()

    def set_properties(self, left_mediators_count=0,
                       right_mediators_count=0,
                       forks_count=0, colliders_count=0):
        self.left_mediators_count = left_mediators_count
        self.right_mediators_count = right_mediators_count
        self.forks_count = forks_count
        self.colliders_count = colliders_count

    def create_a_node(self):
        _node = f'n_{self.get_number_of_nodes()}'
        self.nodes[_node] = Node(name=_node)
        self.add_node_to_graph(_node)
        return _node

    def reset(self):
        for node in self.nodes.keys():
            self.nodes[node].reset()

    def get_total_nodes(self):
        return self.g.number_of_nodes()

    def get_node_names(self):
        return self.g.nodes()

    def add_an_element(self, element='right_mediator'):
        _node = self.get_random_node()
        [x, y, z] = [self.create_a_node() for _ in range(3)]
        if element == 'right_mediator':
            # X -> Y -> Z
            self.add_an_edge_to_graph(x, y)
            self.add_an_edge_to_graph(y, z)
        elif element == 'left_mediator':
            # X <- Y <- Z
            self.add_an_edge_to_graph(z, y)
            self.add_an_edge_to_graph(y, x)
        elif element == 'fork':
            # X <- Y -> Z
            self.add_an_edge_to_graph(y, x)
            self.add_an_edge_to_graph(y, z)
        elif element == 'collider':
            # X -> Y <- Z
            self.add_an_edge_to_graph(x, y)
            self.add_an_edge_to_graph(z, y)
        else:
            print('Unsupported element')

        if _node:
            link = random.sample([x, y, z], 1)[0]
            if random.randint(0, 1):
                self.add_an_edge_to_graph(_node, link)
            else:
                self.add_an_edge_to_graph(link, _node)

    def generate_random_graph(self):
        for _ in range(self.left_mediators_count):
            self.add_an_element(element='left_mediator')

        for _ in range(self.right_mediators_count):
            self.add_an_element(element='right_mediator')

        for _ in range(self.forks_count):
            self.add_an_element(element='fork')

        for _ in range(self.colliders_count):
            self.add_an_element(element='collider')

        return self.get_graph()

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

        def depth_first_traversal(node, g):
            v = self.nodes.get(node).get_value()
            for successor in list(g.successors(node)):
                edge_value = self.get_edge_value(node, successor)
                w, c = edge_value['weight'], edge_value['capacity']
                self.nodes.get(successor).update_value((v * w) + c)
                depth_first_traversal(successor, g)
            return

        """
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
        """

        """
        q = [
        for variable in independent_variables:
            q.append(variable)
            breadth_first_traversal(q, self.g)
        """
        for variable in independent_variables:
            depth_first_traversal(variable)

        return self.nodes.values()

    def run(self, id=0):
        print(f' {id}: Running a breadth first traversal to update node values')


    def get_observations(self, n=0):
        observations = {}
        self.reset()

        # Get all independent variables
        independent_variables = self.get_source_nodes()

        for step in range(n):
            self.generate_observation(independent_variables)
            for node in self.nodes:
                if node in observations.keys():
                    observations[node].append(self.nodes[node].get_value()[0])
                else:
                    observations[node] = [self.nodes[node].get_value()[0]]
            self.reset()
        return pd.DataFrame(observations)


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


def main():
    cg = CausalGraph(nx.DiGraph())
    cg.set_properties(left_mediators_count=1,
                      right_mediators_count=1,
                      forks_count=1,
                      colliders_count=1)
    cg.generate_random_graph()
    observations = cg.get_observations(n=500)
    print(observations.head())

    complete_graph = nx.complete_graph(cg.get_node_names())

    # case a: empty z
    nodes = nPr(cg.get_node_names(), 2)
    for (x, y) in nodes:
        corr = np.corrcoef(observations[x], observations[y])
        corr = corr[0][1]
        # print(x, y)
        # print(f'{x, y} - {corr}')
        if complete_graph.has_edge(x, y) and (np.abs(corr) < 0.9):
            complete_graph.remove_edge(x, y)

    # case a: empty a
    nodes = nPr(cg.get_node_names(), 3)
    for (x, y, z) in nodes:
        # print(f'Partial correlation between {(x, y)} and {z}')
        df = pd.DataFrame({'x': observations[x], 'y': observations[y], 'z': observations[z]})
        result = pg.partial_corr(data=df, x='x', y='y', covar='z').round(3)
        p_value = result['p-val']['pearson']
        r_value = result['r']['pearson']
        if p_value > 0.08:
            # print(f'{x, y} - {p_value} - {r_value}')
            if complete_graph.has_edge(x, y):
                complete_graph.remove_edge(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('Original Graph')
    axes[1].set_title('Predicted Graph')

    draw_graph(g=cg.get_graph(), ax=axes[0])
    draw_graph(g=complete_graph, ax=axes[1])
    # plt.show()

    # sns.scatterplot(x=observations.n_3, y=observations.n_5)
    # sns.scatterplot(x=observations.n_0, y=observations.n_3)
    # sns.scatterplot(x=observations.n_0, y=observations.n_1)
    sns.pairplot(observations)
    plt.show()


if __name__ == "__main__":
    main()