#!/usr/bin/python
# coding: UTF-8
#
# Author: Dawid Laszuk
# Contact: laszukdawid@gmail.com
#
# Last update: 12/03/2017
#
# Feel free to contact for any information.
#
# You can cite this code by referencing:
#   D. Laszuk, "Python implementation of Kuramoto systems," 2017-,
#   [Online] Available: http://www.laszukdawid.com/codes
#
# LICENCE:
# This program is free software on GNU General Public Licence version 3.
# For details of the copyright please see: http://www.gnu.org/licenses/.
"""Based on https://github.com/ethanfetaya/NRI (MIT License)."""

import numpy as np
from scipy.integrate import ode
import numpy as np
import time
import argparse
import os
import json
from Utils import Log
from experiment import Observations
from experiment import Experiment

__version__ = '0.3'
__author__ = 'Dawid Laszuk'

parser = argparse.ArgumentParser()
parser.add_argument('--num_train', type=int, default=2,
                    help='Number of training simulations to generate.')
parser.add_argument('--num_valid', type=int, default=1,
                    help="Number of validation simulations to generate.")
parser.add_argument('--num_test', type=int, default=1,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=10000,
                    help='Length of test set trajectory.')
parser.add_argument('--num_atoms', type=int, default=3,
                    help='Number of atoms (aka time-series) in the simulation.')
parser.add_argument('--lowfreq', type=bool, default=False,
                    help='Signal frequency.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--ode_type', type=str, default="kuramoto",
                    help='Which ODE to use [kuramoto]')
parser.add_argument('--sample_freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--interaction_strength', type=int, default=1,
                    help='Strength of Interactions between particles')
parser.add_argument("--undirected", action="store_true", default=False,
                    help="Have symmetric connections")
parser.add_argument("--n_save_small", type=int, default=100,
                    help="Number of training sequences to save separately.")
parser.add_argument('--data_path', default='/Users/naveenmysore/PycharmProjects/data/')

args = parser.parse_args()


class Kuramoto(Experiment):
    def __init__(self, experiment_id):
        super().__init__(experiment_id)
        self.num_channels = 0
        self.length = 0
        self.sample_freq = 0
        self.num_sim = 0

    def get_num_of_channels(self):
        return self.num_channels

    def set_num_of_channels(self, value):
        self.num_channels = value

    def set_length(self, length):
        self.length = length

    def get_length(self):
        return self.length

    def get_num_of_sim(self):
        return self.num_sim

    def set_num_of_sim(self, value):
        self.num_sim = value

    def get_sample_freq(self):
        return self.sample_freq

    def set_sample_freq(self, sample_freq):
        self.sample_freq = sample_freq

    def set_num_sim(self, num_sim):
        self.num_sim = num_sim

    def _get_channel_vars(self):
        _nc = self.get_num_of_channels()
        column_names = []
        column_names.extend([f'phase_diff_{_id}' for _id in range(_nc)])
        column_names.extend([f'trajectories_{_id}' for _id in range(_nc)])
        column_names.extend([f'phase_{_id}' for _id in range(_nc)])
        column_names.extend([f'intrinsic_freq_{_id}' for _id in range(_nc)])
        return column_names

    def _get_edge_vars(self):
        np = self.get_num_of_channels()
        column_names = []
        for i in range(np):
            for j in range(np):
                if i != j:
                    column_names.append(f'e_{i}_{j}')
        return column_names

    def get_channel_observational_record(self):
        channel_observations = Observations()
        _vars = self._get_channel_vars()
        _vars.append('simulation')
        _vars.append('step')
        channel_observations.set_column_names(columns=_vars)
        return channel_observations

    def get_edge_observational_record(self):
        edge_observations = Observations()
        _vars = self._get_edge_vars()
        _vars.append('simulation')
        _vars.append('step')
        edge_observations.set_column_names(columns=_vars)
        return edge_observations

    def save(self):
        Log.info("Kuramoto", "Settings", f"Saving settings for experiment {self._id}")
        exp_data = dict()
        if os.path.exists(self.experiment_path):
            with open(self.experiment_path) as json_file:
                exp_data = json.load(json_file)
        exp_data[self._id]['settings']['kuramoto']['length'] = self.get_length()
        exp_data[self._id]['settings']['kuramoto']['sample_freq'] = self.get_sample_freq()
        exp_data[self._id]['settings']['kuramoto']['num_channels'] = self.get_num_of_channels()
        exp_data[self._id]['settings']['kuramoto']['num_sim'] = self.get_num_of_sim()
        exp_data[self._id]['settings']['kuramoto']['channel_variables'] = self._get_channel_vars()
        exp_data[self._id]['settings']['kuramoto']['edges_variables'] = self._get_edge_vars()
        with open(self.experiment_path, 'w') as f:
            json.dump(exp_data, f, indent=4)
        Log.info('Kuramoto', 'Settings', f"Saved experiment {self._id} settings to {self.experiment_path}")


class KuramotoSim(object):
    """
    Implementation of Kuramoto coupling model [1] with harmonic terms
    and possible perturbation.
    It uses NumPy and Scipy's implementation of Runge-Kutta 4(5)
    for numerical integration.
    Usage example:
    >>> kuramoto = KuramotoSim(initial_values)
    >>> phase = kuramoto.solve(X)
    [1] Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence
        (Vol. 19). doi: doi.org/10.1007/978-3-642-69689-3
    """

    _noises = { 'logistic': np.random.logistic,
                'normal': np.random.normal,
                'uniform': np.random.uniform,
                'custom': None
              }

    noise_types = _noises.keys()

    def __init__(self, init_values, noise=None):
        """
        Passed arguments should be a dictionary with NumPy arrays
        for initial phase (Y0), intrisic frequencies (W) and coupling
        matrix (K).
        """
        self.dtype = np.float32
        self.dt = 1.
        self.init_phase = np.array(init_values['Y0'])
        self.W = np.array(init_values['W'])
        self.K = np.array(init_values['K'])

        self.n_osc = len(self.W)
        self.m_order = self.K.shape[0]

        self.noise = noise


    @property
    def noise(self):
        """Sets perturbations added to the system at each timestamp.
        Noise function can be manually defined or selected from
        predefined by assgining corresponding name. List of available
        pertrubations is reachable through `noise_types`. """
        return self._noise

    @noise.setter
    def noise(self, _noise):

        self._noise = None
        self.noise_params = None
        self.noise_type = 'custom'

        # If passed a function
        if callable(_noise):
            self._noise = _noise

        # In case passing string
        elif isinstance(_noise, str):

            if _noise.lower() not in self.noise_types:
                self.noise_type = None
                raise NameError("No such noise method")

            self.noise_type = _noise.lower()
            self.update_noise_params(self.dt)

            noise_function = self._noises[self.noise_type]
            self._noise = lambda: np.array([noise_function(**p) for p in self.noise_params])

    def update_noise_params(self, dt):
        self.scale_func = lambda dt: dt/np.abs(self.W**2)
        scale = self.scale_func(dt)

        if self.noise_type == 'uniform':
            self.noise_params = [{'low':-s, 'high': s} for s in scale]
        elif self.noise_type in self.noise_types:
            self.noise_params = [{'loc':0, 'scale': s} for s in scale]
        else:
            pass

    def kuramoto_ODE(self, t, y, arg):
        """General Kuramoto ODE of m'th harmonic order.
           Argument `arg` = (w, k), with
            w -- iterable frequency
            k -- 3D coupling matrix, unless 1st order
            """

        w, k = arg
        yt = y[:,None]
        dy = y-yt
        phase = w.astype(self.dtype)
        if self.noise != None:
            n = self.noise().astype(self.dtype)
            phase += n
        for m, _k in enumerate(k):
            phase += np.sum(_k*np.sin((m+1)*dy),axis=1)

        return phase

    def kuramoto_ODE_jac(self, t, y, arg):
        """Kuramoto's Jacobian passed for ODE solver."""

        w, k = arg
        yt = y[:,None]
        dy = y-yt

        phase = [m*k[m-1]*np.cos(m*dy) for m in range(1,1+self.m_order)]
        phase = np.sum(phase, axis=0)

        for i in range(self.n_osc):
            phase[i,i] = -np.sum(phase[:,i])

        return phase

    def solve(self, t):
        """Solves Kuramoto ODE for time series `t` with initial
        parameters passed when initiated object.
        """
        dt = t[1]-t[0]
        if self.dt != dt and self.noise_type != 'custom':
            self.dt = dt
            self.update_noise_params(dt)

        kODE = ode(self.kuramoto_ODE, jac=self.kuramoto_ODE_jac)
        kODE.set_integrator("dopri5")

        # Set parameters into model
        kODE.set_initial_value(self.init_phase, t[0])
        kODE.set_f_params((self.W, self.K))
        kODE.set_jac_params((self.W, self.K))

        if self._noise != None:
            self.update_noise_params(dt)

        phase = np.empty((self.n_osc, len(t)))

        # Run ODE integrator
        for idx, _t in enumerate(t[1:]):
            phase[:,idx] = kODE.y
            kODE.integrate(_t)

        phase[:,-1] = kODE.y

        return phase


def simulate_kuramoto(num_atoms, num_timesteps=10000, T=None, dt=0.01, undirected=False):
    if T is None:
        # num_timesteps = int((10000 / float(100)) - 1)
        # t0, t1, dt = 0, int((10000 / float(100)) / 10), 0.01
        dt = 0.01
        t0, t1= 0, int(num_timesteps * dt * 10)

        T = np.arange(t0, t1, dt)

    intrinsic_freq = np.random.rand(num_atoms) * 9 + 1.
    initial_phase = np.random.rand(num_atoms) * 2 * np.pi
    edges = np.random.choice(2, size=(num_atoms, num_atoms), p=[0.5, 0.5])
    if undirected:
        # created symmetric edges matrix (i.e. undirected edges)
        edges = np.tril(edges) + np.tril(edges, -1).T
    np.fill_diagonal(edges, 0)

    kuramoto = KuramotoSim({'W': intrinsic_freq,
                         'K': np.expand_dims(edges, 0),
                         'Y0': initial_phase})

    # kuramoto.noise = 'logistic'
    odePhi = kuramoto.solve(T)

    # Subsample
    phase_diff = np.diff(odePhi)[:, ::10] / dt
    trajectories = np.sin(odePhi[:, :-1])[:, ::10]

    # Normalize dPhi (individually)
    min_vals = np.expand_dims(phase_diff.min(1), 1)
    max_vals = np.expand_dims(phase_diff.max(1), 1)
    phase_diff = (phase_diff - min_vals) * 2 / (max_vals - min_vals) - 1

    # Get absolute phase and normalize
    phase = odePhi[:, :-1][:, ::10]
    min_vals = np.expand_dims(phase.min(1), 1)
    max_vals = np.expand_dims(phase.max(1), 1)
    phase = (phase - min_vals) * 2 / (max_vals - min_vals) - 1

    # If oscillator is uncoupled, set trajectory to dPhi to 0 for all t
    isolated_idx = np.where(edges.sum(1) == 0)[0]
    phase_diff[isolated_idx] = 0.

    # Normalize frequencies to [-1, 1]
    intrinsic_freq = (intrinsic_freq - 1.) * 2 / (10. - 1.) - 1.

    phase_diff = np.expand_dims(phase_diff, -1)[:, :num_timesteps, :]
    trajectories = np.expand_dims(trajectories, -1)[:, :num_timesteps, :]
    phase = np.expand_dims(phase, -1)[:, :num_timesteps, :]
    intrinsic_freq = np.expand_dims(np.repeat(
        np.expand_dims(intrinsic_freq, -1),
        num_timesteps, axis=1), -1)

    sim_data = np.concatenate(
        (phase_diff, trajectories, phase, intrinsic_freq),
        -1)

    print("--")
    print(sim_data.shape)
    print("--")

    return sim_data, edges


######################################

def generate_dataset(num_sims, length, sample_freq):
    num_sims = num_sims
    num_timesteps = int((length / float(sample_freq)) - 1)

    t0, t1, dt = 0, int((length / float(sample_freq)) / 10), 0.01
    T = np.arange(t0, t1, dt)

    sim_data_all = []
    edges_all = []
    for i in range(num_sims):
        t = time.time()
        if args.ode_type == "kuramoto":
            sim_data, edges = simulate_kuramoto(args.num_atoms, num_timesteps, T, dt, args.undirected)
            assert sim_data.shape[2] == 4
        else:
            raise Exception("Invalid args.ode_type")

        sim_data_all.append(sim_data)
        edges_all.append(edges)

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))

    data_all = np.array(sim_data_all, dtype=np.float32)
    edges_all = np.array(edges_all, dtype=np.int64)
    print(data_all.shape)

    return data_all, edges_all


if __name__ == "__main__":

    np.random.seed(args.seed)

    suffix = "_" + args.ode_type
    suffix += str(args.num_atoms)

    if args.undirected:
        suffix += "undir"

    if args.interaction_strength != 1:
        suffix += "_inter" + str(args.interaction_strength)

    if args.lowfreq:
        suffix += "_lowfreq"
    print(suffix)

    # NOTE: We first generate all sequences with same length as length_test
    # and then later cut them to required length. Otherwise normalization is
    # messed up (for absolute phase variable).
    print("Generating {} training simulations".format(args.num_train))
    data_train, edges_train = generate_dataset(args.num_train, args.length_test, args.sample_freq)

    print("Generating {} validation simulations".format(args.num_valid))
    data_valid, edges_valid = generate_dataset(args.num_valid, args.length_test, args.sample_freq)

    num_timesteps_train = int((args.length / float(args.sample_freq)) - 1)
    data_train = data_train[:, :, :num_timesteps_train, :]
    data_valid = data_valid[:, :, :num_timesteps_train, :]

    # Save 100 training examples as separate block, so we can compare cLSTM +
    # NRI models.
    small_data_train = data_train[:args.n_save_small]
    small_edges_train = edges_train[:args.n_save_small]

    print("Generating {} test simulations".format(args.num_test))
    data_test, edges_test = generate_dataset(
        args.num_test, args.length_test, args.sample_freq
    )

    savepath = os.path.expanduser(args.data_path)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    print("Saving to {}".format(savepath))
    np.save(
        os.path.join(savepath, "feat_train" + suffix + ".npy"),
        data_train,
    )
    np.save(
        os.path.join(savepath, "edges_train" + suffix + ".npy"),
        edges_train,
    )

    np.save(
        os.path.join(savepath, "feat_train_small" + suffix + ".npy"),
        small_data_train,
    )
    np.save(
        os.path.join(savepath, "edges_train_small" + suffix + ".npy"),
        small_edges_train,
    )

    np.save(
        os.path.join(savepath, "feat_valid" + suffix + ".npy"),
        data_valid,
    )
    np.save(
        os.path.join(savepath, "edges_valid" + suffix + ".npy"),
        edges_valid,
    )

    np.save(
        os.path.join(savepath, "feat_test" + suffix + ".npy"),
        data_test,
    )
    np.save(
        os.path.join(savepath, "edges_test" + suffix + ".npy"),
        edges_test,
    )
