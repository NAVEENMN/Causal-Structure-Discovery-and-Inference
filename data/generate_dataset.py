#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import pandas as pd
import numpy as np
import networkx as nx
import logging
import matplotlib.pyplot as plt
from itertools import permutations
from simulations import spring_particle_system

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def generate_data_schema(save_loc):
    schema = {
        'label': ['positions',
                  'velocity',
                  'edges',
                  'kinetic_energy',
                  'potential_energy',
                  'total_energy'],
        'descriptions': ['Positions of all particles in x and y co ordinates through time',
                         'Velocity of all particles in x and y co ordinates through time',
                         'Causal relationship between particles (spring) through time',
                         'Kinetic energy of all particles through time',
                         'Potential energy of all particles through time',
                         'Total energy of all particles through time'],
        'dimensions': [('total_trajectories', 'total_time', '2', 'num_of_particles'),
                       ('total_trajectories', 'total_time', '2', 'num_of_particles'),
                       ('total_trajectories', 'total_time', 'num_of_particles', 'num_of_particles'),
                       ('total_trajectories', 'total_time', '1'),
                       ('total_trajectories', 'total_time', '1'),
                       ('total_trajectories', 'total_time', '1')]
    }

    df = pd.DataFrame(schema).set_index('label')
    df.to_pickle(f'{save_loc}_schema.pkl')
    print(f"Data schema saved to {save_loc}_schema.pkl")


def generate_data(num_sim, simulation, save_loc='data/simulation_test'):
    print(f"Generating {num_sim} {simulation.get_dynamics()} simulations")
    observations = None
    for i in range(num_sim):
        _, data_frame = simulation.sample_trajectory(total_time_steps=1000, sample_freq=50)
        if observations is None:
            observations = data_frame
        observations = observations.append(data_frame)
    observations.to_csv(f'{save_loc}.csv')
    print(f'Observations of {num_sim} {simulation.get_dynamics()} simulations saved to {save_loc}.csv')


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


class ParticleSystem(GraphStyle):
    def __init__(self, num_of_particles=1, feature_dimension=2):
        super().__init__()
        self.num_of_particles = num_of_particles
        self.spring_constants = np.zeros(shape=(num_of_particles, num_of_particles))
        self.feature_dimension = feature_dimension
        self.dimensions = ['x', 'y']
        self.particle_graph = None
        self.causal_graph = None

    def set_number_of_particles(self, n):
        self.num_of_particles = n
        self.spring_constants = np.zeros(shape=(n, n))

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

    def generate_particle_system_graph(self):
        pass

    def get_causal_graph(self):
        return self.causal_graph

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

    def draw_causal_graph(self, axes):
        axes.set_title('Causal Graph')
        pos = nx.circular_layout(self.causal_graph)
        weights = nx.get_edge_attributes(self.causal_graph, 'weight').values()
        # TODO: Add edge labels
        nx.draw(self.causal_graph, pos,
                with_labels=True,
                width=list(weights),
                node_size=self.get_node_size(),
                node_color=self.get_node_color(),
                font_color=self.get_font_color(),
                ax=axes)


def main():
    #ps = ParticleSystem()
    #ps.set_number_of_particles(n=4)
    #ps.add_spring(pa=0, pb=1, w=0.5)
    #ps.add_spring(pa=0, pb=2, w=1)

    #fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #ps.draw_particle_system_graph(axes=axes[0])
    #ps.draw_causal_graph(axes=axes[1])

    #plt.show()

    _system = spring_particle_system.System(num_particles=3)
    _system.set_number_of_particles(n=3)

    number_of_particles = _system.get_number_particles()
    _system.set_static_edges(edges=np.random.rand(number_of_particles, number_of_particles))
    generate_data(10, _system)

    # Static dynamics uses fixed or static causal graph, periodic uses time varying causal graph.
    #simulation.set_dynamics(dynamics='static')
    #generate_data(args=args, simulation=simulation)w


if __name__ == "__main__":
    main()
