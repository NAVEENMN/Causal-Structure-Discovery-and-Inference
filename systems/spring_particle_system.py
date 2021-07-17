#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
import networkx as nx
from itertools import permutations

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class GraphStyle(object):
    def __init__(self):
        self.node_color = '#0D0D0D'
        self.font_color = '#D9D9D9'
        self.edge_color = '#262626'
        self.node_size = 800

    def get_node_color(self):
        return self.node_color

    def get_font_color(self):
        return self.font_color

    def get_edge_color(self):
        return self.edge_color

    def get_node_size(self):
        return self.node_size


class System(GraphStyle):
    def __init__(self, num_particles=2, min_steps=50, max_steps=200, mode='random'):
        super().__init__()
        self.num_particles = num_particles
        self.interaction_strength = 0.1
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.observations = {}

        self.dynamics = None
        self.mode = mode
        self.static_edges = []
        self.static_init_velocity = []
        self.spring_constants = []

        self.box_size = 5.
        self.loc_std = .5
        self.vel_norm = .5
        self.noise_var = 0.
        self._spring_prob = [0.4, 0.1, 0.0, 0.1, 0.4]
        self._spring_types = np.array([0.0, 0.1, 0.5, 0.8, 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T
        self.feature_dimension = 2

        self.positions = []
        self.velocities = []
        self.edges = []
        self.edge_counter = None
        self.columns = []

        self.dimensions = ['x', 'y']
        self.particle_graph = None
        self.causal_graph = None

    def set_static_edges(self, edges):
        self.spring_constants = edges

    def set_init_velocity(self, init_vel):
        self.static_init_velocity = init_vel

    def set_springs(self, spring_prob, spring_types):
        self._spring_prob = spring_prob
        self._spring_types = spring_types

    def get_number_particles(self):
        return self.num_particles

    def set_number_of_particles(self, n):
        self.num_particles = n

        self.particle_graph = nx.DiGraph()
        self.causal_graph = nx.DiGraph()

        for i in range(n):
            self.particle_graph.add_node(f'p{i}')
            self.causal_graph.add_node(f'p{i}_x')
            self.causal_graph.add_node(f'p{i}_y')

    def add_spring(self, pa, pb, w):
        self.particle_graph.add_node(f'p{pa}')
        self.particle_graph.add_node(f'p{pb}')
        self.spring_constants[pa][pb] = w
        self.spring_constants[pb][pa] = w

        # Assumption: Spring co efficients components are all same in all dimensions
        if self.feature_dimension == 2:
            self.particle_graph.add_edge(f'p{pa}', f'p{pb}', weight=w)
            self.particle_graph.add_edge(f'p{pb}', f'p{pa}', weight=w)

            edges = permutations([f'p{pa}_x', f'p{pa}_y', f'p{pb}_x', f'p{pb}_y'])
            for edge in edges:
                _a = edge[0].replace('_x', '').replace('_y', '')
                _b = edge[1].replace('_x', '').replace('_y', '')
                # Exclude self links
                if _a != _b:
                    self.causal_graph.add_edge(edge[0], edge[1], weight=w)

    def set_dynamics(self, dynamics):
        if dynamics == 'static':
            self.dynamics = 'static'
        elif dynamics == 'periodic':
            self.dynamics = 'periodic'
        else:
            logging.warning(f'Unsupported dynamics {dynamics}')
            logging.info('Setting dynamics to static')
            self.dynamics = 'static'

    def draw_particle_system_graph(self, axes):
        axes.set_title('Particle System')
        weights = nx.get_edge_attributes(self.particle_graph, 'weight').values()
        # TODO: Add edge labels
        nx.draw(self.particle_graph,
                pos=nx.circular_layout(self.particle_graph),
                with_labels=True,
                width=list(weights),
                node_size=self.get_node_size(),
                node_color=self.get_node_color(),
                font_color=self.get_font_color(),
                ax=axes)

    def draw_causal_graph(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[1].set_title('Causal Graph')
        weights = nx.get_edge_attributes(self.causal_graph, 'weight').values()
        # TODO: Add edge labels
        nx.draw(self.causal_graph,
                pos=nx.circular_layout(self.causal_graph),
                with_labels=True,
                width=list(weights),
                node_size=self.get_node_size(),
                node_color=self.get_node_color(),
                font_color=self.get_font_color(),
                ax=axes[1])
        axes[0].set_title('Particle System')
        weights = nx.get_edge_attributes(self.particle_graph, 'weight').values()
        # TODO: Add edge labels
        nx.draw(self.particle_graph,
                pos=nx.circular_layout(self.particle_graph),
                with_labels=True,
                width=list(weights),
                node_size=self.get_node_size(),
                node_color=self.get_node_color(),
                font_color=self.get_font_color(),
                ax=axes[0])
        plt.savefig(f"{os.getcwd()}/media/causal_graph.png")
        plt.clf()

    def get_dynamics(self):
        return self.dynamics

    def get_observations(self):
        return pd.DataFrame(self.observations).set_index('time')

    def _reset(self):
        """
		Reset the simulation to original state
		:return:
		"""
        logging.debug('Resetting simulation')
        self.positions.clear()
        self.velocities.clear()
        self.edges.clear()
        self.columns.clear()
        self.edge_counter = self.get_edge_counter(self.min_steps, self.max_steps + 1)
        self.columns = [f'particle_{i}' for i in range(self.num_particles)]

    def _clamp(self, loc, vel):
        """
		:param loc: 2xN location at one time stamp
		:param vel: 2xN velocity at one time stamp
		:return: location and velocity after hitting walls and returning after
			elastically colliding with walls
		"""
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]

        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def get_edge_counter(self, min_steps, max_steps):
        counter = np.random.choice(list(range(min_steps, max_steps)), size=(self.num_particles, self.num_particles))
        counter = np.tril(counter) + np.tril(counter, -1).T
        np.fill_diagonal(counter, 0)
        return counter

    def get_init_pos_velocity(self):
        """
        This function samples position and velocity from a distribution.
		These position and velocity will be used as
		initial position and velocity for all particles.
		:return: initial position and velocity
        :return:
        """
        init_position = np.random.randn(2, self.num_particles) * self.loc_std
        init_velocity = self.static_init_velocity if self.mode == 'manual' else np.random.randn(2, self.num_particles)

        # Compute magnitude of this velocity vector and format to right shape
        v_norm = np.linalg.norm(init_velocity, axis=0)

        # Scale by magnitude
        init_velocity = init_velocity * self.vel_norm / v_norm

        return init_position, init_velocity

    def get_force(self, _edges, current_positions):
        """
		:param _edges: Adjacency matrix representing mutual causality
		:param current_positions: current coordinates of all particles
		:return: net forces acting on all particles.
		"""
        force_matrix = - self.interaction_strength * _edges
        np.fill_diagonal(force_matrix, 0)
        x_cords, y_cords = current_positions[0, :], current_positions[1, :]
        x_diffs = np.subtract.outer(x_cords, x_cords).reshape(1, self.num_particles, self.num_particles)
        y_diffs = np.subtract.outer(y_cords, y_cords).reshape(1, self.num_particles, self.num_particles)
        force_matrix = force_matrix.reshape(1, self.num_particles, self.num_particles)
        _force = (force_matrix * np.concatenate((x_diffs, y_diffs))).sum(axis=-1)
        _force[_force > self._max_F] = self._max_F
        _force[_force < -self._max_F] = -self._max_F
        return _force

    def generate_edges(self):
        """
		This function generates causality graph where particles are treated as nodes.
		:return: causality graph represented as edges where particles
		"""
        if self.mode == 'manual':
            _edges = self.static_edges
        else:
            # Sample nxn springs _spring_types which each holding a probability spring_prob
            _edges = np.random.choice(self._spring_types, size=(self.num_particles, self.num_particles),
                                      p=self._spring_prob)

        # Establish symmetry causal interaction
        _edges = np.tril(_edges) + np.tril(_edges, -1).T

        # Nullify self interaction or causality
        np.fill_diagonal(_edges, 0)

        return _edges

    def make_observation(self, current_position, velocity):
        if 'time' not in self.observations:
            self.observations['time'] = [int(time.time())]
        else:
            self.observations['time'].append(int(time.time()))

        _edges = self.spring_constants
        for m in range(len(_edges)):
            for n in range(len(_edges[0])):
                if f's_p{m}_p{n}' not in self.observations:
                    self.observations[f's_p{m}_p{n}'] = [_edges[m][n]]
                else:
                    self.observations[f's_p{m}_p{n}'].append(_edges[m][n])

        for dim in range(len(velocity)):
            for pid in range(len(velocity[0])):
                if dim == 0:
                    if f'v_p{pid}_xdim' not in self.observations:
                        self.observations[f'v_p{pid}_xdim'] = [velocity[dim][pid]]
                    else:
                        self.observations[f'v_p{pid}_xdim'].append(velocity[dim][pid])
                else:
                    if f'v_p{pid}_ydim' not in self.observations:
                        self.observations[f'v_p{pid}_ydim'] = [velocity[dim][pid]]
                    else:
                        self.observations[f'v_p{pid}_ydim'].append(velocity[dim][pid])

        for dim in range(len(current_position)):
            for pid in range(len(current_position[0])):
                if dim == 0:
                    if f'p_p{pid}_xdim' not in self.observations:
                        self.observations[f'p_p{pid}_xdim'] = [current_position[dim][pid]]
                    else:
                        self.observations[f'p_p{pid}_xdim'].append(current_position[dim][pid])
                else:
                    if f'p_p{pid}_ydim' not in self.observations:
                        self.observations[f'p_p{pid}_ydim'] = [current_position[dim][pid]]
                    else:
                        self.observations[f'p_p{pid}_ydim'].append(current_position[dim][pid])

    def sample_trajectory(self, total_time_steps=10000, sample_freq=10):

        # Reset simulation
        self._reset()

        # Data frame columns and index for particles
        # columns = [f'particle_{i}' for i in range(self.num_particles)]
        index = ['x_cordinate', 'y_cordinate']

        # Initialize causality between particles.
        _edges = self.generate_edges()

        # Initialize the first position and velocity from a distribution
        init_position, init_velocity = self.get_init_pos_velocity()

        # Adding initial position and velocity of particles to trajectory.
        init_position, init_velocity = self._clamp(init_position, init_velocity)
        _position = pd.DataFrame(init_position, columns=self.columns, index=index)
        _velocity = pd.DataFrame(init_velocity, columns=self.columns, index=index)
        _edge = pd.DataFrame(_edges, columns=self.columns, index=self.columns)
        self.positions.append(_position)
        self.velocities.append(_velocity)
        self.edges.append(_edge)

        # Compute initial forces between particles.
        init_force_between_particles = self.get_force(_edges, init_position)

        # Compute new velocity.
        '''
		F = m * (dv/dt), for unit mass
		dv = dt * F
		velocity - current_velocity = dt * F
		velocity = current_velocity + (self._delta_T * F)
		'''
        get_velocity = lambda initial_velocity, forces: initial_velocity + (self._delta_T * forces)

        velocity = get_velocity(init_velocity, init_force_between_particles)
        current_position = init_position

        edges_counter = self.edge_counter

        observations = {}

        for i in range(1, total_time_steps):

            # Compute new position based on current velocity and positions.
            new_position = current_position + (self._delta_T * velocity)
            new_position, velocity = self._clamp(new_position, velocity)

            # Adding new position and velocity of particles to trajectory.
            if i % sample_freq == 0:
                _position = pd.DataFrame(new_position, columns=self.columns, index=index)
                _velocity = pd.DataFrame(velocity, columns=self.columns, index=index)
                _edge = pd.DataFrame(_edges, columns=self.columns, index=self.columns)
                self.positions.append(_position)
                self.velocities.append(_velocity)
                self.edges.append(_edge)

            # If causal graph is periodic, flip causal edges when counter turns zero
            if self.dynamics == 'periodic':
                edges_counter -= 1
                change_mask = np.where(edges_counter == 0, 1, 0)
                if np.any(change_mask):
                    new_edges = np.where(_edges == 0, 1.0, 0)
                    _edges = np.where(change_mask == 1, new_edges, _edges)
                    new_counter = self.get_edge_counter(self.min_steps, self.max_steps + 1)
                    edges_counter = np.where(change_mask == 1, new_counter, edges_counter)

            # Compute forces between particles
            force_between_particles = self.get_force(_edges, new_position)

            # Compute new velocity based on current velocity and forces between particles.
            new_velocity = velocity + (self._delta_T * force_between_particles)

            # Update velocity and position
            velocity = new_velocity
            current_position = new_position

            # Add noise to observations
            current_position += np.random.randn(2, self.num_particles) * self.noise_var
            velocity += np.random.randn(2, self.num_particles) * self.noise_var

            if i % sample_freq:
                self.make_observation(current_position, velocity)

        # Compute energy of the system
        kinetic_energies, potential_energies, total_energies = self.get_energy()

        # construct data frame
        trajectory = {
            'time': int(time.time()),
            'positions': self.positions,
            'velocity': self.velocities,
            'edges': self.edges,
            'kinetic_energy': kinetic_energies,
            'potential_energy': potential_energies,
            'total_energy': total_energies,
        }


        return pd.DataFrame(trajectory)

    def get_energy(self):
        '''
		Total Energy = Kinetic Energy (K) + Potential Energy (U)
		Kinetic Energy (K) = (1/2) * m * velocity^2 : unit mass m
		Potential Energy (U) = m * g * h: h is distance, g is field, unit mass m
		:return: energy
		'''

        # Compute Kinetic Energy for each snap shot
        # Kinetic energy = (1/2) m * v^2, here assume a unit mass
        ek = lambda velocity: 0.5 * (velocity ** 2).sum(axis=0)
        kinetic_energies = [ek(_velocities) for _velocities in self.velocities]
        kinetic_energies = [pd.DataFrame({'kinetic_energy': ke, 'particles': self.columns}).set_index('particles') for
                            ke in kinetic_energies]

        # Compute Potential Energy at each snap shot
        # potential energy = m * g * d, here assume a unit mass
        # g represents interaction strength and h represents distance.
        potential_energies = []
        for time_step, position in enumerate(self.positions):
            _pos = position.T.to_numpy()
            _u = []
            for particle_index in range(0, self.num_particles):
                particle_pos = position[f'particle_{particle_index}'].T.to_numpy()
                position_fill_mat = np.full(_pos.shape, particle_pos)
                distances = np.sqrt(np.square(position_fill_mat - _pos).sum(axis=1))
                pe = np.dot(self.edges[time_step][f'particle_{particle_index}'], distances ** 2)
                _u.append(0.5 * self.interaction_strength * pe)
            potential_energies.append(
                pd.DataFrame({'potential_energy': _u, 'particles': self.columns}).set_index('particles'))

        # Compute total energy of the system
        total_energies = []
        for time_step in range(len(potential_energies)):
            total_en = kinetic_energies[time_step]['kinetic_energy'] + potential_energies[time_step]['potential_energy']
            total_energies.append(
                pd.DataFrame({'total_energy': total_en, 'particles': self.columns}).set_index('particles'))
        kinetic_energies = [_ken.T for _ken in kinetic_energies]
        potential_energies = [_pen.T for _pen in potential_energies]
        total_energies = [_ten.T for _ten in total_energies]
        return kinetic_energies, potential_energies, total_energies


def generate_data(num_sim):
    print(f"Generating {num_sim} simulations")
    observations = None
    for i in range(num_sim):
        num_particles = 3
        sim = System(num_particles=3, min_steps=500, max_steps=1000)
        sim.set_number_of_particles(n=3)
        sim.set_static_edges(edges=np.random.randn(num_particles, num_particles))
        _ = sim.sample_trajectory(total_time_steps=1000, sample_freq=50)
        data_frame = sim.get_observations()
        if observations is None:
            observations = data_frame
        observations = observations.append(data_frame)
    observations.to_csv(os.path.join(os.getcwd(), 'data', 'simulation.csv'))
    print(f'Observations of {num_sim} simulations saved to data/simulation.csv')


if __name__ == '__main__':
    generate_data(num_sim=2000)
