import numpy as np
from scipy.linalg import svdvals, svd
import tigramite
from tigramite import data_processing as pp


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
        mean, sd = np.random.normal(0, 1, 1), 1
        self.nodes[_node] = (mean, sd)
        return _node

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
            logging.info('Adding  x->y->z')
            self.g.add_edge(x, y, color=color_code, weight=random.uniform(0, 1))
            self.g.add_edge(y, z, color=color_code, weight=random.uniform(0, 1))
        else:
            logging.info('Adding  x<-y<-z')
            self.g.add_edge(z, y, color=color_code, weight=random.uniform(0, 1))
            self.g.add_edge(y, x, color=color_code, weight=random.uniform(0, 1))

        if node:
            link = random.sample([x, y, z], 1)[0]
            if random.randint(0, 1):
                self.g.add_edge(node, link, color=color_code, weight=random.uniform(0, 1))
            else:
                self.g.add_edge(link, node, color=color_code, weight=random.uniform(0, 1))

    # x <- y -> z
    def _add_a_fork(self):
        color_code = 'b'

        node = None
        if self.g.number_of_nodes() != 0:
            node = random.sample(self.g.nodes(), 1)[0]

        x = self.create_a_node()
        y = self.create_a_node()
        z = self.create_a_node()

        logging.info('Adding  x<-y->z')
        self.g.add_edge(y, x, color=color_code, weight=random.uniform(0, 1))
        self.g.add_edge(y, z, color=color_code, weight=random.uniform(0, 1))
        if node:
            link = random.sample([x, y, z], 1)[0]
            if random.randint(0, 1):
                self.g.add_edge(node, link, color=color_code, weight=random.uniform(0, 1))
            else:
                self.g.add_edge(link, node, color=color_code, weight=random.uniform(0, 1))

    # x -> y <- z
    def _add_a_collider(self):
        color_code = 'b'

        node = None
        if self.g.number_of_nodes() != 0:
            node = random.sample(self.g.nodes(), 1)[0]

        x = self.create_a_node()
        y = self.create_a_node()
        z = self.create_a_node()

        logging.info('Adding  x->y<-z')
        self.g.add_edge(x, y, color=color_code, weight=random.uniform(0, 1))
        self.g.add_edge(z, y, color=color_code, weight=random.uniform(0, 1))
        if node:
            link = random.sample([x, y, z], 1)[0]
            if random.randint(0, 1):
                self.g.add_edge(node, link, color=color_code, weight=random.uniform(0, 1))
            else:
                self.g.add_edge(link, node, color=color_code, weight=random.uniform(0, 1))

    def generate_random_graph(self, m=1, f=1, c=1):
        for _ in range(m):
            self._add_a_mediator()
        for _ in range(f):
            self._add_a_fork()
        for _ in range(c):
            self._add_a_collider()

    def get_edges(self):
        print(nx.get_edge_attributes(self.g, 'weight'))
        return self.g.edges()

    def show_graph(self):
        edges = self.g.edges()
        colors = nx.get_edge_attributes(self.g, 'color').values()
        weights = nx.get_edge_attributes(self.g, 'weight').values()
        pos = nx.circular_layout(self.g)
        nx.draw(self.g, pos,
                with_labels=True,
                edge_color=list(colors),
                width=list(weights),
                node_size=500)
        plt.show()

    def _update_graph(self, _nodes):
        _edges = nx.get_edge_attributes(self.g, 'weight')
        for _edge in _edges:
            # a -> b
            (a, b) = _edge
            _nodes[b] += (_edges[_edge] * _nodes[a])
        return _nodes

    def get_observation(self, n=0):
        observations = {}
        # initialize values to nodes
        sample_var = lambda _mu, _sd: np.random.normal(_mu, _sd, 1)
        _nodes = {node: sample_var(self.nodes[node][0], self.nodes[node][1]) for node in self.nodes}
        for step in range(n):
            _nodes = self._update_graph(_nodes)
            for node in self.nodes:
                if node in observations.keys():
                    observations[node].append(_nodes[node][0])
                else:
                    observations[node] = [_nodes[node][0]]
        return pd.DataFrame(observations)


class CausalDiscovery(object):
    def __init__(self):
        self.num_of_variables = 0
        self._variables = [f'$X^{i}$'for i in range(self.num_of_variables*2)]
        self._data = None
        self.data_frame = None

    def set_num_of_variables(self, num_of_variables=0):
        self.num_of_variables = num_of_variables
        self._variables = [f'$X^{i}$'for i in range(num_of_variables)]

    def get_sample_observations(self, t=1000):
        np.random.seed(42)
        links_coeffs = {0: [((0, -1), 0.7), ((1, -1), -0.8)],
                        1: [((1, -1), 0.8), ((3, -1), 0.8)],
                        2: [((2, -1), 0.5), ((1, -2), -0.5), ((3, -3), 0.6)],
                        3: [((3, -1), 0.4)]}
        _data, true_parents_neighbours = pp.var_process(links_coeffs, T=t)
        self._set_variables(variables=[f'$X^{i}$'for i in range(len(links_coeffs))])
        self.set_data(_data)
        return self.data_frame

    def _set_variables(self, variables):
        self._variables = variables

    def set_data(self, _data):
        self._data = _data
        self.data_frame = pp.DataFrame(self._data, datatime=np.arange(len(self._data)), var_names=self._variables)
        return self.data_frame

    def get_variables(self):
        return self._variables

    def plot_time_series(self):
        from tigramite import plotting as tp
        tp.plot_timeseries(dataframe=self.data_frame)

    def pcmci(self, conditional_independence_test):
        from tigramite.pcmci import PCMCI
        return PCMCI(dataframe=self.data_frame, cond_ind_test=conditional_independence_test, verbosity=1)



def pca_eigenvals(d):
    """
    Compute the eigenvalues of the covariance matrix of the data d. the covariance matrix is computed as d * d^T.
    """
    # remove mean of each row
    d = d - np.mean(d, axis=1)[:, np.newaxis]
    return 1.0/(d.shape[1] - 1) * svdvals(d)**2


def pca_eigenvals_gf(d):
    """
    Compute the PCA for a geo-field that will be unrolled into one dimension.
    axis[0] must be time, other axes are considered spatial
    and will be unrolled so that the PCA is performed on a 2D matrix.
    """
    # reshape by combining all spatial dimensions
    # np.prod(d.shape[1:]) -> (lat * long)
    d = np.reshape(d, (d.shape[0], np.prod(d.shape[1:])))
    # we need the constructed single spatial dimension to be on axis 0
    d = d.transpose()
    return d



def conditional_independence_test(dataframe):
    from tigramite.pcmci import  PCMCI
    from tigramite.independence_tests import ParCorr
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)
    results = pcmci.run_pcmci(tau_max=2, pc_alpha=None)
    return results

