#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cmath
import logging
import numpy as np
import pandas as pd
import networkx as nx
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


class GraphStyle(object):
    def __init__(self):
        self.node_color = '#0D0D0D'
        self.font_color = '#D9D9D9'
        self.edge_color = '#262626'
        self.node_size = 800

    def get_node_color(self):
        return self.node_color


class Environment(object):
    def __init__(self):
        self.box_size = 5.0
        self._delta_T = 0.001
        self.dimensions = 2
        self._positions = []
        self._velocities = []

    def reset(self):
        self._positions.clear()
        self._velocities.clear()

    def add_a_particle(self):
        pass

    def get_positions(self):
        return self._positions

    def get_velocities(self):
        return self._velocities


class System(Environment, GraphStyle):
    def __init__(self):
        super().__init__()
        self.min_steps = 50
        self.max_steps = 200
        self.particle_graph = nx.DiGraph()
        self.causal_graph = nx.DiGraph

    def draw(self):
        pass


class SpringSystem(System):
    def __init__(self):
        super().__init__()
        self.k = []
        self.num_particles = 0
        self.observations = {}

    def _add_a_particle(self):
        _particle_name = f'p_{self.particle_graph.number_of_nodes()}'
        self.particle_graph.add_node(_particle_name)

    def add_particles(self, num_of_particles=0):
        logging.debug(f'Creating a spring particle system with {num_of_particles} particles')
        for _ in range(num_of_particles):
            self._add_a_particle()
        self.num_particles = self.particle_graph.number_of_nodes()
        logging.info(f'Created a spring particle system with {num_of_particles} particles')

    def _add_a_spring(self, particle_a, particle_b, spring_constant=0.0):
        if self.num_particles == 0:
            logging.error('Environment has no particles to add a spring')
            return
        if spring_constant != 0:
            logging.debug(f'Adding spring between {particle_a} and {particle_b} with k={spring_constant} ')
            self.particle_graph.add_edge(f'p{particle_a}', f'p{particle_b}', weight=spring_constant)

    def add_springs(self, spring_constants_matrix):

        if self.num_particles == 0:
            logging.error('Environment has no particles to add a spring')
            return

        if spring_constants_matrix.shape != (self.num_particles, self.num_particles):
            logging.error('Shapes of spring constants matrix and number of particles wont match')
            return

        # Establish symmetry
        spring_constants_matrix = np.tril(spring_constants_matrix) + np.tril(spring_constants_matrix, -1).T

        # Nullify self interaction or causality
        np.fill_diagonal(spring_constants_matrix, 0)

        self.k = spring_constants_matrix

        for i in range(self.num_particles):
            for j in range(self.num_particles):
                self._add_a_spring(particle_a=i, particle_b=j, spring_constant=self.k[i][j])

        logging.info(f'Added springs to a spring particle system')

    def simulate(self, total_time_steps=1000, sample_freq=10):
        if self.particle_graph.number_of_nodes() == 0:
            logging.warning('Nothing to simulate, add particles')
            return

        def get_init_pos_velocity():
            """
            This function samples position and velocity from a distribution.
            These position and velocity will be used as
            initial position and velocity for all particles.
            :return: initial position and velocity
            """
            loc_std = 0.5
            init_position = np.random.randn(2, self.num_particles) * loc_std
            init_velocity = np.random.randn(2, self.num_particles)

            # Compute magnitude of this velocity vector and format to right shape
            v_norm = np.linalg.norm(init_velocity, axis=0)

            # Scale by magnitude
            vel_norm = 0.5
            init_velocity = init_velocity * vel_norm / v_norm

            return init_position, init_velocity

        def make_an_observation(current_position, velocity):
            noise_var = 0.05

            # Add noise to observations
            current_position += np.random.randn(2, self.num_particles) * noise_var
            velocity += np.random.randn(2, self.num_particles) * noise_var

            if 'time' not in self.observations:
                self.observations['time'] = [int(time.time())]
            else:
                self.observations['time'].append(int(time.time()))

            _edges = self.k
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

        def get_force(_edges, current_positions):
            """
            :param _edges: Adjacency matrix representing mutual causality
            :param current_positions: current coordinates of all particles
            :return: net forces acting on all particles.
            F = -kx. The proportional constant k is called the spring constant.
            It is a measure of the spring's stiffness.
            When a spring is stretched or compressed, so that its length changes by an amount x from
            its equilibrium length, then it exerts a force F = -kx in a direction towards its equilibrium position.
            Assumption here is initially all particles were at origin, now some particles are moved stretching spring.
            TODO: Re verify this force computation
            """
            spring_constants = _edges
            np.fill_diagonal(spring_constants, 0)
            x_cords, y_cords = current_positions[0, :], current_positions[1, :]
            x_diffs = np.square(np.subtract.outer(x_cords, x_cords))
            y_diffs = np.square(np.subtract.outer(y_cords, y_cords))
            distance_matrix = np.sqrt(np.square(x_diffs) + np.square(y_diffs))
            # F = -k*x, by hooke's law
            force_matrix = -1 * spring_constants * distance_matrix
            # [net_force on particle 0, net_force on particle 1, net_force on particle 2, ..]
            net_force_matrix = np.sum(force_matrix, axis=1)
            return net_force_matrix

        # Initialize the first position and velocity from a distribution
        init_position, init_velocity = get_init_pos_velocity()

        '''
        Compute initial forces between particles.
        F = -k * dx, by hooke's law
        '''
        init_force_between_particles = get_force(self.k, init_position)

        '''
        Compute new velocity.
        dv = dt * F
        velocity - current_velocity = dt * F
        velocity_x = current_velocity + (self._delta_T * F * cos(theta))
        velocity_y = current_velocity + (self._delta_T * F * sin(theta))
        velocity = (velocity_x+velocity_y) + (self._delta_T * F)
        '''
        pos_vec = init_position / np.linalg.norm(init_position)
        theta = np.arccos(np.dot(pos_vec.T, pos_vec))
        velocity_x = init_velocity + (self._delta_T * init_force_between_particles)
        velocity_y = init_velocity + (self._delta_T * init_force_between_particles)

        # Initialize current position
        current_position = init_position

        for i in range(1, total_time_steps):
            # Compute new position based on current velocity and positions.
            new_position = current_position + (self._delta_T * velocity)
            # Compute forces between particles
            force_between_particles = get_force(self.k, new_position)
            # Compute new velocity based on current velocity and forces between particles.
            new_velocity = velocity + (self._delta_T * force_between_particles)
            # Update velocity and position
            velocity = new_velocity
            current_position = new_position
            if i % sample_freq == 0:
                make_an_observation(current_position, velocity)

    def get_observations(self):
        return pd.DataFrame(data=self.observations).set_index('time')

def main():
    sp = SpringSystem()
    sp.add_particles(num_of_particles=3)
    spring_constants_matrix = np.asarray([[0, 0, 1],
                                          [0, 0, 0],
                                          [1, 0, 0]])
    sp.add_springs(spring_constants_matrix=spring_constants_matrix)
    sp.simulate(total_time_steps=1000, sample_freq=10)
    df = sp.get_observations()
    print(df.head())


if __name__ == "__main__":
    main()
