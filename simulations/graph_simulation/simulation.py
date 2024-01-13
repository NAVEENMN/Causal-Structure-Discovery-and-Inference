from graph import Graph
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

class Simulation:
    def __init__(self):
        self.df = None
        self.gh = None
        self.dataframe = None
        self.observations = None

    def add_graph(self, graph):
        self.gh = graph
        self.dataframe = self.gh.get_readings()

    def add_observation(self, observation):
        if self.observations is None:
            self.observations = observation
        else:
            self.observations = pd.concat([self.observations, observation], ignore_index=True)

    def get_readings(self):
        return self.dataframe

    def update_readings(self):
        self.gh.update_nodes()
        self.dataframe = self.gh.get_readings()

    def reinit_graph(self):
        self.gh.reinit()

    def snapshot(self):
        self.gh.print()

    def reindex(self):
        self.observations = self.observations.set_index('timestamp')

    def print_observations(self):
        print(self.observations)

    def save_graph(self):
        self.gh.print()

    def save_observations(self):
        dir_name = str(len(list(os.walk('../../data/simulatory_observations/')))+1)
        # TODO: Add number of nodes and simulation length.
        os.mkdir(f"../../data/simulatory_observations/{dir_name}")

        self.plot_combined_graph_and_timeseries(path=f"../../data/simulatory_observations/{dir_name}/")
        self.observations.to_csv(f"../../data/simulatory_observations/{dir_name}/observations.csv")

    def plot_combined_graph_and_timeseries(self, path):
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # Adjust figsize as needed

        # Plot the graph on the top subplot
        axs[0].set_title('Original Graph')
        pos = nx.circular_layout(self.gh.gh)  # You can choose a different layout if you prefer
        nx.draw(self.gh.gh, pos, ax=axs[0], node_color='r', edge_color='b', with_labels=True)

        # Plot the time series on the bottom subplot
        df = self.observations
        axs[1].set_title('Observations')
        for column in df.columns:
            axs[1].plot(df.index, df[column], label=column)
        axs[1].set_xlabel('Timestamp')
        axs[1].set_ylabel('Value')
        axs[1].legend()
        axs[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        # plt.show()
        plt.savefig(path)

    def plot_time_series(self):
        df = self.observations
        plt.figure(figsize=(10, 6))  # Set the figure size as desired

        # Plot each node column
        for column in df.columns:
            plt.plot(df.index, df[column], label=column)

        plt.xlabel('Timestamp')  # Set the x-axis label
        plt.ylabel('Value')  # Set the y-axis label
        plt.title('Time Series Plot')  # Set the title of the plot
        plt.legend()  # Show legend to identify the node columns
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

        # Display the plot
        plt.show()

def main():
    simulation = Simulation()
    graph = Graph(num_of_nodes=6)
    graph.generate_random_graph()
    simulation.add_graph(graph=graph)

    for _ in range(30):
        simulation.reinit_graph()
        simulation.update_readings()
        observation = simulation.get_readings()
        simulation.add_observation(observation)

    simulation.reindex()
    simulation.save_graph()
    simulation.print_observations()
    simulation.save_observations()


if __name__ == "__main__":
    main()