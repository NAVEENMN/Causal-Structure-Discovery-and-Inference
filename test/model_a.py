import numpy as np
import pandas as pd
import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr

# Load data
schema = pd.read_pickle('../data/data_schema.pkl')
data = pd.read_pickle('../data/dyari.pkl')
train = data.sample(frac=0.8, random_state=200)
test = data.drop(train.index)

# Print data schema
print(schema)

# All positions
simulation_sample = train.iloc[0]
positions = simulation_sample.trajectories.positions

time_step = 0
print(positions[time_step])
num_of_particles = positions[time_step].shape[1]

snapshot = np.asarray(positions[time_step]).flatten()
print(snapshot)

data = []
for time_step in range(20):
    snapshot = np.asarray(positions[time_step]).flatten()
    data.append(np.asarray(snapshot))
data = np.asarray(data)

variable_names = [f'$X^{i}$'for i in range(num_of_particles*2)]
print(variable_names)
dataframe = pp.DataFrame(data, datatime=np.arange(len(data)), var_names=variable_names)
print(dataframe)
