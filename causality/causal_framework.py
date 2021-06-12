import numpy as np
from scipy.linalg import svdvals, svd
import tigramite
from tigramite import data_processing as pp


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


def gen_t_data():
    # [((variable_id, time_step), weight)]
    link_coeffs = {
        0: [((0, -1), 0.7), ((1, -1), -0.8)],
        1: [((1, -1), 0.8), ((3, -1), 0.8)],
        2: [((2, -1), 0.5), ((1, -2), 0.5), ((3, -3), 0.6)],
        3: [((3, -1), 0.4)]
    }
    data, true_parents_neighbours = pp.var_process(link_coeffs, T=100)
    T, N = data.shape

    var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$']
    dataframe = pp.DataFrame(data, datatime=np.arange(len(data)), var_names=var_names)
    return dataframe


def conditional_independence_test(dataframe):
    from tigramite.pcmci import  PCMCI
    from tigramite.independence_tests import ParCorr
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)
    results = pcmci.run_pcmci(tau_max=2, pc_alpha=None)
    return results


data = gen_t_data()
print(data.values)
print(conditional_independence_test(dataframe=data))