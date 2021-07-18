import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def load_spring_particle_data(time_steps=500):
    # Load data
    schema = pd.read_pickle('../data/simulation_test_schema.pkl')
    data = pd.read_pickle('../data/simulation_test.pkl')
    train = data.sample(frac=0.8, random_state=200)
    # test = data.drop(train.index)

    data = []
    graphs = []

    # All positions
    for simulation_id in range(5):

        simulation_sample = train.iloc[simulation_id]
        positions = simulation_sample.trajectories.positions
        edges = simulation_sample.trajectories.edges
        for time_step in reversed(range(time_steps)):
            try:
                snapshot = positions[time_step] - positions[time_step-1]
                snapshot = np.asarray(snapshot).flatten()
                # print(edges[time_step])
                data.append(snapshot)
                graphs.append(edges)
            except KeyError as e:
                continue

    data = np.asarray(data)
    return data, graphs, schema


def nPr(_set, r):
    from itertools import permutations
    return list(permutations(_set, r))


def partial_correlation(data, x, y, z):
    import pingouin as pg
    df = pd.DataFrame({'x': data[x], 'y': data[y], 'z': data[z]})
    result = pg.partial_corr(data=df, x='x', y='y', covar='z').round(3)
    return result


def draw_graph(graph, axes=None):
    weights = nx.get_edge_attributes(graph, 'weight').values()
    pos = nx.circular_layout(graph)
    nx.draw(graph, pos,
            with_labels=True,
            width=list(weights),
            node_size=500,
            ax=axes)


def main():
    data, graphs, schema = load_spring_particle_data()
    print(graphs)
    print("  ** Data Schema  **  ")
    print(schema)

    # Normalize the data
    print(data.shape)
    #print(data)
    #print(np.mean(data, axis=0))
    #print(np.std(data, axis=0))
    data = (data-np.mean(data, axis=0)) / np.std(data, axis=0)
    #print(data)
    #exit()
    observations = data #preprocessing.normalize(data)

    # Setup column names as a preparation for causal discovery
    col_names = [f'n_{i}' for i in range(observations.shape[1])]
    observations = pd.DataFrame(data=data)
    observations.columns = col_names
    print(observations.head())

    # create a complete graph with these variables
    complete_graph = nx.complete_graph(col_names)

    # case a: When conditioning set (z) is empty
    nodes = nPr(col_names, 2)
    for (x, y) in nodes:
        corr = np.corrcoef(observations[x], observations[y], rowvar=False)
        corr = corr[0][1]
        if complete_graph.has_edge(x, y) and (np.abs(corr) < 0.9):
            complete_graph.remove_edge(x, y)

    # case b: When conditioning set (z) is not empty
    nodes = nPr(col_names, 3)
    for (x, y, z) in nodes:
        # print(f'Partial correlation between {(x, y)} and {z}')
        result = partial_correlation(observations, x, y, z)
        p_value, r_value = result['p-val']['pearson'],  result['r']['pearson']
        if p_value > 0.08:
            # print(f'{x, y} - {p_value} - {r_value}')
            if complete_graph.has_edge(x, y):
                complete_graph.remove_edge(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('Original Graph')
    axes[1].set_title('Predicted Graph')

    draw_graph(complete_graph, axes=axes[1])
    sns.pairplot(observations)

    plt.show()

if __name__ == "__main__":
    main()
