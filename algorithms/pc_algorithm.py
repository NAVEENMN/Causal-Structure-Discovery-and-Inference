import numpy as np
import networkx as nx
import pandas as pd
import pingouin as pg
from utils import Utils
from itertools import permutations
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def partial_correlation(x, y, z):
    """
    The p-value roughly indicates the probability of an uncorrelated system producing datasets
    that have a Pearson correlation at least as extreme as the one computed from these datasets.
    :param
    """

    def _standardize(_array):
        _array = np.asarray(_array)
        _mean = np.mean(_array)
        _std = np.std(_array)
        return np.reshape((_array-_mean)/_std, (-1, 1))

    # standardize
    _x = _standardize(x)
    _y = _standardize(y)
    _z = _standardize(z)

    def regress_and_get_residuals(_input, target):
        logging.info(f'Performing linear regression')
        reg = LinearRegression().fit(_input, target)
        logging.info(f'Computing residuals for target')
        predictions = np.reshape(reg.predict(_input), (-1, 1))
        residuals = np.subtract(target, predictions).flatten()
        return residuals

    residuals_x_z = regress_and_get_residuals(_input=_z, target=_x)
    residuals_y_z = regress_and_get_residuals(_input=_z, target=_y)

    # calculate pearson coefficient
    # TODO: Plot residuals and see how they look
    residuals_x_z = np.reshape(residuals_x_z, (-1,))
    residuals_y_z = np.reshape(residuals_y_z, (-1,))
    coeff, p_value = pearsonr(residuals_x_z, residuals_y_z)
    result = {'p_value': p_value, 'r': coeff}
    return result


class PC(object):
    def __init__(self, p_value, graph, observations):
        self.p_value = p_value
        self.graph = graph
        self.observations = observations

    def run(self):
        step = 0
        node_names = self.graph.get_node_names()

        complete_graph = nx.complete_graph(node_names)

        # case a: empty z
        nodes = permutations(node_names, 2)
        for (x, y) in nodes:
            a, b = np.reshape(self.observations[x].to_numpy(), (-1,)), \
                   np.reshape(self.observations[y].to_numpy(), (-1,))
            coeff, p_value = pearsonr(a, b)
            logging.info(f'{x, y} - {p_value}')
            #corr = np.corrcoef(a, b, rowvar=False)
            #corr = corr[0][1]
            if complete_graph.has_edge(x, y) and (np.abs(p_value) > self.p_value):
                # If observed p value is greater than threshold (self.p_value)
                # We have stronger confidence in rejecting the null hypothesis
                # null hypothesis is that there exists a link between x and y
                complete_graph.remove_edge(x, y)

            G = nx.Graph()
            G.add_nodes_from(node_names)
            G.add_edge(x, y)
            step += 1
            Utils.save_graph(self.graph,
                             G,
                             complete_graph,
                             step,
                             attr=[f'Pcorr({x}-{y})', f"{format(p_value, '.2f')} ? {self.p_value}"])

        Z = self.graph.get_all_parents()
        logging.info(f'ancestors {Z}')
        # case b: z
        nodes = permutations(node_names, 3)
        for (x, y, z) in nodes:

            # we need conditioning set to contain only ancestors
            if z not in Z:
                logging.info(f'Skipping {z}')
                continue

            # print(f'Partial correlation between {(x, y)} and {z}')
            result = partial_correlation(self.observations[x],
                                         self.observations[y],
                                         self.observations[z])
            # result = pg.partial_corr(data=df, x='x', y='y', covar='z').round(3)
            p_value, r_value = result['p_value'], result['r']
            logging.info(f'{x, y} | {z} - {p_value} - {r_value}')
            if np.abs(p_value) > self.p_value:
                # print(f'{x, y} - {p_value} - {r_value}')
                # If observed p value is greater than threshold (self.p_value)
                # We have stronger confidence in rejecting the null hypothesis
                # null hypothesis is that there doesnt exists a link between x and y
                logging.info(f'{p_value}, {self.p_value} ')
                if complete_graph.has_edge(x, y):
                    complete_graph.remove_edge(x, y)

            G = nx.Graph()
            G.add_nodes_from(node_names)
            G.add_edge(x, y)
            G.add_edge(y, z)

            step += 1
            Utils.save_graph(self.graph,
                             G,
                             complete_graph,
                             step,
                             [f'Pcorr({x}-{y}|{z})', f"{format(p_value, '.2f')} ? {self.p_value}"])
