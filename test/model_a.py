import numpy as np
import pandas as pd
import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr
from causality import CausalDiscovery

# Load data
schema = pd.read_pickle('../data/simulation_test_schema.pkl')
data = pd.read_pickle('../data/simulation_test.pkl')
train = data.sample(frac=0.8, random_state=200)
test = data.drop(train.index)

# All positions
simulation_sample = train.iloc[0]
positions = simulation_sample.trajectories.positions

data = []
for time_step in range(20):
    snapshot = np.asarray(positions[time_step]).flatten()
    data.append(np.asarray(snapshot))
data = np.asarray(data)


cd = CausalDiscovery()
cd.set_num_of_variables(4*2)
data_frame = cd.set_data(data)
print(cd.get_variables())

parcorr = ParCorr(significance='analytic')
pcmci = PCMCI(dataframe=data_frame, cond_ind_test=parcorr, verbosity=1)
results = pcmci.run_pcmci(tau_max=2, pc_alpha=None)
print(results)