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

    def simulate(self):
        if self.particle_graph.number_of_nodes() == 0:
            logging.warning('Nothing to simulate, add particles')
            return

        num_particles = self.particle_graph.number_of_nodes()

        def get_init_pos_velocity():
            """
            This function samples position and velocity from a distribution.
            These position and velocity will be used as
            initial position and velocity for all particles.
            :return: initial position and velocity
            """
            loc_std = 0.5
            vel_norm = 0.5
            init_position = np.random.randn(2, num_particles) * loc_std
            init_velocity = np.random.randn(2, num_particles)

            # Compute magnitude of this velocity vector and format to right shape
            v_norm = np.linalg.norm(init_velocity, axis=0)

            # Scale by magnitude
            init_velocity = init_velocity * vel_norm / v_norm

            return init_position, init_velocity

        def get_force(_edges, current_positions):
            """
            :param _edges: Adjacency matrix representing mutual causality
            :param current_positions: current coordinates of all particles
            :return: net forces acting on all particles.
            TODO: Re verify this force computation
            """
            force_matrix = -1 * _edges
            np.fill_diagonal(force_matrix, 0)
            x_cords, y_cords = current_positions[0, :], current_positions[1, :]
            x_diffs = np.subtract.outer(x_cords, x_cords)
            y_diffs = np.subtract.outer(y_cords, y_cords)
            force_matrix = force_matrix.reshape(1, self.num_particles, self.num_particles)
            v = np.concatenate(x_diffs, y_diffs)
            _force = (force_matrix * np.concatenate((x_diffs, y_diffs))).sum(axis=-1)
            return _force

        # Initialize the first position and velocity from a distribution
        init_position, init_velocity = get_init_pos_velocity()

        # Compute initial forces between particles.
        init_force_between_particles = get_force(self.k, init_position)

        print(init_force_between_particles)


def main():
    sp = SpringSystem()
    sp.add_particles(num_of_particles=3)
    spring_constants_matrix = np.asarray([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    sp.add_springs(spring_constants_matrix=spring_constants_matrix)
    sp.simulate()


if __name__ == "__main__":
    main()
