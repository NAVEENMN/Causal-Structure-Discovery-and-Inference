from graph import Graph
import pandas as pd


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

    def print_observations(self):
        print(self.observations)

    def save_observations(self):
        self.observations.to_csv("../../data/observations.csv")


def main():
    simulation = Simulation()
    graph = Graph(num_of_nodes=5)
    graph.generate_random_graph()
    simulation.add_graph(graph=graph)

    for _ in range(5):
        simulation.reinit_graph()
        simulation.update_readings()
        observation = simulation.get_readings()
        simulation.add_observation(observation)

    simulation.print_observations()
    simulation.save_observations()

if __name__ == "__main__":
    main()