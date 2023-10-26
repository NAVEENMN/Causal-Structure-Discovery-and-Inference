#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Gather's and simulates different data for causal discovery.
writes observational data and schema to /data
"""
import os
import time
import argparse
import numpy as np
from scipy import io
from experiment import Experiment
from netsim import NetSim
from springs import Spring
from springs import SpringSim
from kuramoto import Kuramoto
from kuramoto import KuramotoSim
from Utils import Log

parser = argparse.ArgumentParser()
parser.add_argument('--n-vars', type=int, default=5,
                    help='Number of balls/particles/channels in the simulation.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--num-train', type=int, default=10,
                    help='Number of training simulations to generate.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--source', type=str, default='/Users/naveenmysore/Documents/data/netsim/source/sim3.mat',
                    help='path where data is downloaded.')
parser.add_argument('--save_path', type=str, default='/Users/naveenmysore/Documents/data/',
                    help='path where data is downloaded.')
parser.add_argument('--min_change_step', type=int, default=1000,
                    help='minimum step of changing interaction for spring simulations')
parser.add_argument('--max_change_step', type=int, default=1000,
                    help='maximum step of changing interaction spring simulations')

args = parser.parse_args()

# Create a new experiment
experiment = Experiment()
experiment.create()


def gather_kuramoto_data():
    def simulate_kuramoto(sim_id, channel_observations, edges_observations, num_atoms, num_timesteps=10000, T=None, dt=0.01, undirected=False):
        if T is None:
            # num_timesteps = int((10000 / float(100)) - 1)
            # t0, t1, dt = 0, int((10000 / float(100)) / 10), 0.01
            dt = 0.01
            t0, t1 = 0, int(num_timesteps * dt * 10)

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

        sim_data = np.concatenate((phase_diff, trajectories, phase, intrinsic_freq), -1)
        observation = dict()
        _edges = dict()
        observation['simulation'] = sim_id
        _edges['simulation'] = sim_id
        for _id, channel in enumerate(sim_data):
            for time_step, values in enumerate(channel):
                observation['step'] = time_step
                _edges['step'] = time_step
                observation[f'phase_diff_{_id}'] = values[0]
                observation[f'trajectories_{_id}'] = values[1]
                observation[f'phase_{_id}'] = values[2]
                observation[f'intrinsic_freq_{_id}'] = values[3]
                for si in range(args.n_vars):
                    for sj in range(args.n_vars):
                        if si != sj:
                            _edges[f'e_{si}_{sj}'] = edges[si][sj]

        channel_observations.add_an_observation(observation)
        edges_observations.add_an_observation(_edges)

        return sim_data, edges

    # *** Control Variables***
    _kuramoto = Kuramoto(experiment_id=experiment.get_id())
    _kuramoto.set_num_of_channels(args.n_vars)
    _kuramoto.set_length(args.length)
    _kuramoto.set_sample_freq(sample_freq=args.sample_freq)
    _kuramoto.set_num_sim(num_sim=args.num_train)
    _kuramoto.save()
    # ******** Run simulation ***

    channel_observations = _kuramoto.get_channel_observational_record()
    edges_observations = _kuramoto.get_edge_observational_record()

    num_timesteps = int((args.length / float(args.sample_freq)) - 1)

    t0, t1, dt = 0, int((args.length / float(args.sample_freq)) / 10), 0.01
    T = np.arange(t0, t1, dt)
    sim_data_all = []
    edges_all = []
    for i in range(args.num_train):
        t = time.time()
        sim_data, edges = simulate_kuramoto(i, channel_observations, edges_observations, args.n_vars, num_timesteps, T, dt, False)
        """
        if args.ode_type == "kuramoto":
            sim_data, edges = simulate_kuramoto(i, args.num_vars, num_timesteps, T, dt, args.undirected)
            assert sim_data.shape[2] == 4
        else:
            raise Exception("Invalid args.ode_type")
        """
        sim_data_all.append(sim_data)
        edges_all.append(edges)

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))

    data_all = np.array(sim_data_all, dtype=np.float32)
    edges_all = np.array(edges_all, dtype=np.int64)

    channel_observations.save_observations(path=args.save_path + 'kuramoto',
                                            name=f'observations_{experiment.get_id()}')
    edges_observations.save_observations(path=args.save_path + 'kuramoto',
                                          name=f'edges_{experiment.get_id()}')


def gather_net_sim_data():
    def add_entries(simulation, _data, _observations, _edgeobs):
        subject = 0
        for i, values in enumerate(data['ts'], 1):
            if i % int(data['Ntimepoints']) == 0:
                subject += 1
            _data = dict()
            _edata = dict()
            _data['subject'] = subject
            _edata['subject'] = subject
            for channel_id, entry in enumerate(values):
                _data['step'] = i
                _edata['step'] = i
                _data[f'channel_{channel_id}'] = entry
                for si in range(args.n_vars):
                    for sj in range(args.n_vars):
                        if si != sj:
                            _edata[f'e_{si}_{sj}'] = data['net'][subject-1][si][sj]
            _observations.add_an_observation(_data)
            _edgeobs.add_an_observation(_edata)

    if not os.path.exists(args.source):
        print('Download data from source https://www.fmrib.ox.ac.uk/datasets/netsim/')
        exit()

    # *** Control Variables***
    _netsim = NetSim(experiment_id=experiment.get_id())
    data = io.loadmat(args.source)
    _netsim.set_num_of_subjects(int(data['Nsubjects']))
    _netsim.set_num_of_channels(int(data['Nnodes']))
    _netsim.set_length(int(data['Ntimepoints']))
    _netsim.save()
    # ********
    simulation_id = os.path.basename(args.source)
    _id = simulation_id.split('.')[0]
    observations = _netsim.get_channel_observational_record()
    edge_observations = _netsim.get_edge_observational_record()
    add_entries(simulation=_id, _data=data, _observations=observations, _edgeobs = edge_observations)
    observations.save_observations(path=args.save_path+'netsim', name=f'observations_{experiment.get_id()}')
    observations.save_observations(path=args.save_path + 'netsim', name=f'edges_{experiment.get_id()}')


def gather_spring_data():
    # *** Control Variables***
    _springs = Spring(experiment_id=experiment.get_id())
    _springs.set_numb_of_particles(num_of_particles=args.n_vars)
    _springs.set_traj_length(traj_length=args.length)
    _springs.set_sample_freq(sample_freq=args.sample_freq)
    _springs.set_period(period=args.max_change_step)
    _springs.set_num_sim(num_sim=args.num_train)
    _springs.save()
    # ******** Run simulation ***
    sim = SpringSim(noise_var=0.0, n_balls=args.n_vars)
    particle_observations = _springs.get_particle_observational_record()
    spring_observations = _springs.get_springs_observational_record()
    for i in range(args.num_train):
        t = time.time()
        loc, vel, edges = sim.sample_trajectory_dynamic(T=args.length,
                                                        sample_freq=args.sample_freq,
                                                        min_step=args.min_change_step,
                                                        max_step=args.max_change_step)
        for ind, lc in enumerate(loc):
            x_positions, y_positions = lc[0], lc[1]
            x_vel, y_vel = vel[ind][0], vel[ind][1]
            observation = dict()
            observation[f'trajectory_step'] = f'{i}_{ind}'
            for particle_id in range(args.n_vars):
                observation[f'p_{particle_id}_x_position'] = x_positions[particle_id]
                observation[f'p_{particle_id}_y_position'] = y_positions[particle_id]
                observation[f'p_{particle_id}_x_velocity'] = x_vel[particle_id]
                observation[f'p_{particle_id}_y_velocity'] = y_vel[particle_id]

            sp_observation = dict()
            sp_observation[f'trajectory_step'] = f'{i}_{ind}'
            for si in range(args.n_vars):
                for sj in range(args.n_vars):
                    if si != sj:
                            sp_observation[f'e_{si}_{sj}'] = edges[ind][si][sj]

            particle_observations.add_an_observation(observation)
            spring_observations.add_an_observation(sp_observation)

        if i % 100 == 0:
            Log.info("Springs", "Run", f"Iter: {i}, Simulation time: {time.time() - t}")
    # *** Save observations
    Log.info("Springs", "Saving", f"observations.")
    particle_observations.save_observations(path=args.save_path+'springs',
                                            name=f'observations_{experiment.get_id()}')
    spring_observations.save_observations(path=args.save_path+'springs',
                                          name=f'edges_{experiment.get_id()}')
    Log.info("Springs", "Simulation", f"complete.")


def setup_environment():
    """
    Step directory and files needed prior to simulations
    :return:
    """
    Log.info("Simulation", "Run", f"Setting up environment.")
    def create_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)
            Log.info("Simulation", "Run", f"created {path}")
    create_dir(os.path.join(os.getcwd(), 'data'))
    create_dir(os.path.join(os.getcwd(), 'media'))
    create_dir(os.path.join(os.getcwd(), 'results'))
    create_dir(os.path.join(os.getcwd(), 'meta'))


if __name__ == "__main__":
    #setup_environment()
    gather_kuramoto_data()
    gather_net_sim_data()
    gather_spring_data()
