#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import argparse

import numpy as np
import pandas as pd
from simulations import spring_particle_system

parser = argparse.ArgumentParser()
parser.add_argument('--num-sim', type=int, default=10,
                    help='Number of simulations to generate.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--sample-freq', type=int, default=10,
                    help='How often to sample the trajectory.')
parser.add_argument('--n-particles', type=int, default=3,
                    help='Number of particles in the simulation.')


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


def main():
    args = parser.parse_args()
    simulation = spring_particle_system.System(num_particles=args.n_particles)
    # Static dynamics uses fixed or static causal graph, periodic uses time varying causal graph.
    simulation.set_dynamics(dynamics='static')
    generate_data(args=args, simulation=simulation)


if __name__ == "__main__":
    main()
