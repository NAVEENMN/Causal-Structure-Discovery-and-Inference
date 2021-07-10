#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import pandas as pd
import numpy as np
import networkx as nx
import logging
import matplotlib.pyplot as plt
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


def generate_data(args, simulation, save_loc='data/simulation_test'):

    print(f"Generating {args.num_sim} {simulation.get_dynamics()} simulations")
    trajectories = []
    particles = []
    dynamics = []

    for i in range(args.num_sim):
        t = time.time()
        data_frame = simulation.sample_trajectory(total_time_steps=args.length, sample_freq=args.sample_freq)
        trajectories.append(data_frame)
        # Number of particles are fixed for a given simulation for the current model.
        particles.append(args.n_particles)
        dynamics.append(simulation.get_dynamics())
        print(f"Simulation {i}, time: {time.time() - t}")

    data = {
        'trajectories': trajectories,
        'particles': particles,
        'dynamics': dynamics,
        'simulation_id': [f'simulation_{i}' for i in range(args.num_sim)]
    }
    df = pd.DataFrame(data).set_index('simulation_id')
    df.to_pickle(f'{save_loc}.pkl')
    print(f"Simulations saved to {save_loc}.pkl")
    generate_data_schema(save_loc)
    print("Creating gif for last trajectory.")
    simulation.create_gif()


class ParticleSystem(object):
    def __init__(self, num_of_particles=1, feature_dimension=2):
        self.num_of_particles = num_of_particles
        self.spring_constants = np.zeros(shape=(num_of_particles, num_of_particles))
        self.feature_dimension = feature_dimension
        self.dimensions = ['x', 'y']

        self.particle_graph = nx.DiGraph()
        self.causal_graph = nx.DiGraph()

    def set_number_of_particles(self, n):
        self.num_of_particles = n
        self.spring_constants = np.zeros(shape=(n, n))

    def add_spring(self, pa, pb, w):
        self.spring_constants[pa][pb] = w
        self.spring_constants[pb][pa] = w

        # Assumption: Spring co efficients components are all same in all dimensions
        if self.feature_dimension == 2:
            self.particle_graph.add_edge(f'p{pa}', f'p{pb}', weight=w)
            self.particle_graph.add_edge(f'p{pb}', f'p{pa}', weight=w)
            self.causal_graph.add_edge(f'p{pa}_x', f'p{pb}_y', weight=w)
            self.causal_graph.add_edge(f'p{pb}_y', f'p{pa}_x', weight=w)

    def generate_particle_system_graph(self):
        pass

    def get_causal_graph(self):
        return self.causal_graph

    def draw_particle_system_graph(self, axes):
        axes.set_title('Particle System')
        pos = nx.circular_layout(self.particle_graph)
        weights = nx.get_edge_attributes(self.particle_graph, 'weight').values()
        nx.draw(self.particle_graph, pos,
                with_labels=True,
                width=list(weights),
                node_size=500,
                ax=axes)

    def draw_causal_graph(self, axes):
        axes.set_title('Causal Graph')
        pos = nx.circular_layout(self.causal_graph)
        weights = nx.get_edge_attributes(self.causal_graph, 'weight').values()
        nx.draw(self.causal_graph, pos,
                with_labels=True,
                width=list(weights),
                node_size=500,
                ax=axes)


def main():
    ps = ParticleSystem()
    ps.set_number_of_particles(n=3)
    ps.add_spring(pa=0, pb=1, w=0.5)
    ps.add_spring(pa=0, pb=2, w=1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ps.draw_particle_system_graph(axes=axes[0])
    ps.draw_causal_graph(axes=axes[1])

    plt.show()

    #simulation = spring_particle_system.System(num_particles=3)
    # Static dynamics uses fixed or static causal graph, periodic uses time varying causal graph.
    #simulation.set_dynamics(dynamics='static')
    #generate_data(args=args, simulation=simulation)


if __name__ == "__main__":
    main()
