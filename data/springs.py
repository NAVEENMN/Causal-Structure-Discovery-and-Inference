#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
This script generates spring particle system data.
"""
import os
import json
from Utils import Log
import numpy as np
from experiment import Observations
from experiment import Experiment


class Spring(Experiment):
    def __init__(self, experiment_id):
        super().__init__(experiment_id)
        self.num_of_particles = 0
        self.tau_min = 1
        self.tau_max = 10
        self.min_step = 0
        self.max_step = 0
        self.traj_length = 0
        self.sample_freq = 0
        self.num_sim = 0
        self.period = 0
        self.init_vel = 0.0
        self.particle_vars = []
        self.spring_vars = []
        self.p_threshold = 0.05

    def load_settings(self):
        _id = self.get_id()
        exp_data = {}
        if os.path.exists(self.experiment_path):
            with open(self.experiment_path) as json_file:
                exp_data = json.load(json_file)
        self.set_tau_min(exp_data[self._id]['settings']['tau_min'])
        self.set_tau_max(exp_data[self._id]['settings']['tau_max'])
        self.set_traj_length(exp_data[self._id]['settings']['traj_length'])
        self.set_num_sim(exp_data[self._id]['settings']['num_sim'])
        self.set_sample_freq(exp_data[self._id]['settings']['sample_freq'])
        self.set_period(exp_data[self._id]['settings']['period'])
        self.set_initial_velocity(exp_data[self._id]['settings']['initial_velocity'])
        self.set_numb_of_particles(exp_data[self._id]['settings']['number_of_particles'])
        self.particle_vars = exp_data[self._id]['variables']['particles']
        self.spring_vars = exp_data[self._id]['variables']['springs']

    def load_results(self):
        with open(self.experiment_path) as json_file:
            exp_data = json.load(json_file)
            if exp_data[self.get_id()]['results']['conducted']:
                self.set_p_threshold(exp_data[self.get_id()]['results']['p_threshold'])
                self.set_auroc(exp_data[self.get_id()]['results']['auroc'])
            else:
                Log.error("Simulation", "Experiment", "This experiment is not conducted")

    def get_vars(self):
        return self.particle_vars, self.spring_vars

    def set_tau_min(self, tau_min):
        self.tau_min = tau_min

    def get_tau_min(self):
        return self.tau_min

    def set_tau_max(self, tau_max):
        self.tau_max = tau_max

    def set_min_step(self, min_step):
        self.min_step = min_step

    def set_max_step(self, max_step):
        self.max_step = max_step

    def get_min_step(self):
        return self.min_step

    def get_max_step(self):
        return self.max_step

    def get_tau_max(self):
        return self.tau_max

    def set_traj_length(self, traj_length):
        self.traj_length = traj_length

    def get_traj_length(self):
        return self.traj_length

    def set_sample_freq(self, sample_freq):
        self.sample_freq = sample_freq

    def get_sample_freq(self):
        return self.sample_freq

    def set_num_sim(self, num_sim):
        self.num_sim = num_sim

    def get_num_of_sim(self):
        return self.num_sim

    def set_numb_of_particles(self, num_of_particles):
        self.num_of_particles = num_of_particles

    def get_numb_of_particles(self):
        return self.num_of_particles

    def set_period(self, period):
        self.period = period

    def get_period(self):
        return self.period

    def set_initial_velocity(self, vel):
        self.init_vel = vel

    def get_initial_velocity(self):
        return self.init_vel

    def set_p_threshold(self, p_th):
        self.p_threshold = p_th

    def get_p_threshold(self):
        return self.p_threshold

    def set_auroc(self, auroc):
        self.auroc = auroc

    def get_auroc(self):
        return self.auroc

    def _get_particle_vars(self):
        np = self.get_numb_of_particles()
        column_names = []
        column_names.extend([f'p_{particle_id}_x_position' for particle_id in range(np)])
        column_names.extend([f'p_{particle_id}_y_position' for particle_id in range(np)])
        column_names.extend([f'p_{particle_id}_x_velocity' for particle_id in range(np)])
        column_names.extend([f'p_{particle_id}_y_velocity' for particle_id in range(np)])
        return column_names

    def get_particle_observational_record(self):
        particle_observations = Observations()
        _vars = self._get_particle_vars()
        _vars.append('trajectory_step')
        particle_observations.set_column_names(columns=_vars)
        return particle_observations

    def _get_springs_vars(self):
        np = self.get_numb_of_particles()
        column_names = []
        for i in range(np):
            for j in range(np):
                if i != j:
                    column_names.append(f'e_{i}_{j}')
        return column_names

    def get_springs_observational_record(self):
        spring_observations = Observations()
        _vars = self._get_springs_vars()
        _vars.append('trajectory_step')
        spring_observations.set_column_names(columns=_vars)
        return spring_observations

    def load_recent(self):
        meta_data = dict()
        if os.path.exists(self.meta_path):
            with open(self.meta_path) as json_file:
                meta_data = json.load(json_file)
        exp_id = meta_data['recent']
        self.set_id(exp_id)
        self.load_settings()

    def save(self):
        Log.info("Springs", "Settings", f"Saving settings for experiment {self._id}")
        exp_data = dict()
        if os.path.exists(self.experiment_path):
            with open(self.experiment_path) as json_file:
                exp_data = json.load(json_file)
        exp_data[self._id]['settings']['springs']['min_step'] = self.get_min_step()
        exp_data[self._id]['settings']['springs']['max_step'] = self.max_step
        exp_data[self._id]['settings']['springs']['tau_min'] = self.get_tau_min()
        exp_data[self._id]['settings']['springs']['tau_max'] = self.get_tau_max()
        exp_data[self._id]['settings']['springs']['traj_length'] = self.get_traj_length()
        exp_data[self._id]['settings']['springs']['sample_freq'] = self.get_sample_freq()
        exp_data[self._id]['settings']['springs']['period'] = self.get_period()
        exp_data[self._id]['settings']['springs']['num_sim'] = self.get_num_of_sim()
        exp_data[self._id]['settings']['springs']['number_of_particles'] = self.get_numb_of_particles()
        exp_data[self._id]['settings']['springs']['initial_velocity'] = self.get_initial_velocity()
        exp_data[self._id]['settings']['springs']['particle_variables'] = self._get_particle_vars()
        exp_data[self._id]['settings']['springs']['edge_variables'] = self._get_springs_vars()

        with open(self.experiment_path, 'w') as f:
            json.dump(exp_data, f, indent=4)
        Log.info('Springs', 'Settings', f"Saved experiment {self._id} settings to {self.experiment_path}")

    def publish_results(self, p_threshold, aurocs, tau_min, tau_max):
        import numpy as np
        # TODO: Compute numpy results at source not here.
        with open(self.experiment_path) as json_file:
            exp = json.load(json_file)
        exp[self.get_id()]['results']['conducted'] = True
        exp[self.get_id()]['results']['tau'] = tau_max
        exp[self.get_id()]['results']['p_threshold'] = p_threshold
        exp[self.get_id()]['results']['auroc']['mean'] = np.mean(aurocs)
        exp[self.get_id()]['results']['auroc']['max'] = np.max(aurocs)
        exp[self.get_id()]['results']['auroc']['min'] = np.min(aurocs)
        exp[self.get_id()]['results']['auroc']['std'] = np.std(aurocs)

        # update settings used
        exp[self.get_id()]['settings']['tau_min'] = tau_min
        exp[self.get_id()]['settings']['tau_max'] = tau_max

        with open(self.experiment_path, 'w') as f:
            json.dump(exp, f, indent=4)


class SpringSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=.5, vel_norm=.5,
                 interaction_strength=.1, noise_var=0.):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._spring_types = np.array([0., 0.5, 1.])
        self._delta_T = 0.01
        self._max_F = 0.1 / self._delta_T

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] * (dist ** 2) / 2
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_trajectory_dynamic(self, T=10000, sample_freq=10,
                                  spring_prob=[1. / 2, 0, 1. / 2],
                                  min_step=50, max_step=200):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0

        # Sample edges
        def get_tri_matrix(m):
            m = np.tril(m) + np.tril(m, -1).T
            np.fill_diagonal(m, 0)
            return m

        edges = np.random.choice(self._spring_types,
                                 size=(self.n_balls, self.n_balls),
                                 p=spring_prob)
        edges = get_tri_matrix(edges)

        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        edges_collection = np.zeros((T_save, n, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)
        edges_collection[0, :, :] = edges
        edges_counter = np.random.choice(list(range(min_step, max_step + 1)),
                                         size=(self.n_balls, self.n_balls))
        edges_counter = get_tri_matrix(edges_counter)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            forces_size = - self.interaction_strength * edges
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)
                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    edges_collection[counter, :, :] = edges
                    counter += 1
                # use edge_counter to count whether the edge should be changed,
                # if counter is reduced to zero, then flip the edge
                edges_counter -= 1
                change_mask = np.where(edges_counter == 0, 1, 0)
                if (np.any(change_mask)):
                    # flip the edges
                    new_edges = np.where(edges == 0, 1., 0.)
                    # new_edges = get_tri_matrix(new_edges)
                    edges = np.where(change_mask == 1, new_edges, edges)
                    new_counter = np.random.choice(list(range(min_step, max_step + 1)),
                                                   size=(self.n_balls, self.n_balls))
                    new_counter = get_tri_matrix(new_counter)
                    edges_counter = np.where(change_mask == 1, new_counter, edges_counter)
                forces_size = - self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, edges_collection

    def sample_trajectory(self, T=10000, sample_freq=10,
                          spring_prob=[1. / 2, 0, 1. / 2],
                          fixed_edges=False):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        if fixed_edges:
            edges = np.load('meta/fixed_edges.npy')
        else:
            edges = np.random.choice(self._spring_types,
                                     size=(self.n_balls, self.n_balls),
                                     p=spring_prob)
        edges = np.tril(edges) + np.tril(edges, -1).T
        np.fill_diagonal(edges, 0)
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            forces_size = - self.interaction_strength * edges
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = - self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, edges



if __name__ == '__main__':
    sim = SpringSim()
    # sim = ChargedParticlesSim()

    t = time.time()
    loc, vel, edges = sim.sample_trajectory(T=5000, sample_freq=100)

    print(edges)
    print("Simulation time: {}".format(time.time() - t))
    vel_norm = np.sqrt((vel ** 2).sum(axis=1))
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([-5., 5.])
    axes.set_ylim([-5., 5.])
    for i in range(loc.shape[-1]):
        plt.plot(loc[:, 0, i], loc[:, 1, i])
        plt.plot(loc[0, 0, i], loc[0, 1, i], 'd')
    plt.figure()
    energies = [sim._energy(loc[i, :, :], vel[i, :, :], edges) for i in
                range(loc.shape[0])]
    plt.plot(energies)
    plt.show()