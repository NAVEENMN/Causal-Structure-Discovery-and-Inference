import numpy as np
import pandas as pd
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

# Load data
schema = pd.read_pickle('../data/simulation_test_schema.pkl')
data = pd.read_pickle('../data/simulation_test.pkl')
train = data.sample(frac=0.8, random_state=200)
test = data.drop(train.index)

# All positions
simulation_sample = train.iloc[0]
positions = simulation_sample.trajectories.positions

data = []
for time_step in reversed(range(20)):
    snapshot = np.asarray(positions[time_step]).flatten()
    data.append(np.asarray(snapshot[:2]))
data = np.asarray(data)
print(data)

logging.info(f"Data of shape {data.shape} loaded")


class CausalDiscovery(object):
    def __init__(self):
        self.num_of_variables = 0
        self._variables = [f'$X^{i}$'for i in range(self.num_of_variables*2)]
        self._data = None
        self.data_frame = None

    def _set_num_of_variables(self, num_of_variables=0):
        self.num_of_variables = num_of_variables
        self._variables = [f'$X^{i}$'for i in range(num_of_variables)]
        logging.debug(f"Variables set : {self._variables}")

    def get_sample_observations(self, t=1000):
        np.random.seed(42)
        links_coeffs = {0: [((0, -1), 0.7), ((1, -1), -0.8)],
                        1: [((1, -1), 0.8), ((3, -1), 0.8)],
                        2: [((2, -1), 0.5), ((1, -2), -0.5), ((3, -3), 0.6)],
                        3: [((3, -1), 0.4)]}
        _data, true_parents_neighbours = pp.var_process(links_coeffs, T=t)
        self._set_num_of_variables(num_of_variables=4)
        self.set_data(_data)
        return self.data_frame

    def set_data(self, _data):
        self._data = _data
        self._set_num_of_variables(num_of_variables=_data.shape[1])
        self.data_frame = pp.DataFrame(self._data, datatime=np.arange(len(self._data)), var_names=self._variables)
        return self.data_frame

    def get_variables(self):
        return self._variables

    def get_data_frame(self):
        return self.data_frame

    def plot_time_series(self):
        from tigramite import plotting as tp
        tp.plot_timeseries(dataframe=self.data_frame)

    def pcmci(self, conditional_independence_test):
        from tigramite.pcmci import PCMCI
        return PCMCI(dataframe=self.data_frame, cond_ind_test=conditional_independence_test, verbosity=1)

    def run_conditional_independence_test(self):
        from tigramite.pcmci import PCMCI
        parcorr = ParCorr(significance='analytic')
        pcmci = PCMCI(dataframe=self.data_frame, cond_ind_test=parcorr, verbosity=3)
        results = pcmci.run_pcmci(tau_max=2, pc_alpha=None)
        return results


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


cd = CausalDiscovery()
data_frame = cd.set_data(data)
results = cd.run_conditional_independence_test()
print(results)