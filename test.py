import os
import torch
import random
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt

time_slice = 4
particles = 5
features = 5
trajectory_length = 1000
sample_freq = 10

PATH = os.path.dirname(os.path.realpath(__file__))
print(PATH)
saved_path = os.path.join(PATH, 'saved')

class Data:
    def __init__(self, batch_size=2, observation_length=20):
        self.data = pd.read_pickle('dyari.pkl')
        self.train_data = self.data.sample(frac=0.8, random_state=200)
        self.test_data = self.data.drop(self.train_data.index)
        self.batch_size = batch_size
        self.observation_length = observation_length
        self.trajectory_length = trajectory_length

    def get_batch(self, mode='train'):
        batch = self.train_data if mode == 'train' else self.test_data

        # Sample a random trajectories
        trajectory_ids = random.sample(range(0, len(batch)), self.batch_size)
        observations = []
        for _id in trajectory_ids:
            positions = batch.iloc[_id].trajectories.positions
            # random observation point
            index = random.randint(0, (len(positions)-(self.observation_length+2)))
            # make an observation
            observation = positions.iloc[index:index+self.observation_length+1]
            observations.append(observation)

        _reshape = lambda _x: np.reshape(_x.to_numpy(), (1, particles * 2))
        batching = dict()
        for i, obs in enumerate(observations):
            #print(f'observation {i}')
            for time_stamp in range(self.observation_length+1):
                #print(obs.iloc[time_stamp])
                if time_stamp not in batching:
                    batching[time_stamp] = []
                batching[time_stamp].append(_reshape(obs.iloc[time_stamp]))

        X = [np.reshape(np.asarray(batching[time_stamp]), (self.batch_size, 1, particles*2)) for time_stamp in range(self.observation_length)]
        Y = np.reshape(np.asarray(batching[self.observation_length]), (self.batch_size, 1, particles*2))
        return X, Y


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input dim is 8, output dim is 8
        self.lstm = nn.LSTM(particles*2, particles*2)
        # initialize the hidden state.
        self.hidden = (torch.randn(1, 1, particles*2), torch.randn(1, 1, particles*2))

    def forward(self, x, batch_size):
        x = torch.Tensor(x)
        for i in x:
            # Step through the sequence one element at a time.
            # after each step, hidden contains the hidden state.
            out, self.hidden = self.lstm(i.view(batch_size, 1, -1), self.hidden)
        return out


class Model(Net):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.parameters(),
                                         lr=0.01,
                                         momentum=0.9)

    def loss(self, y_prediction, y_real):
        y_real = torch.Tensor(y_real)
        loss = self.criterion(y_prediction, y_real)
        return loss

    def print_params(self, x):
        for param in self.parameters():
            print(param)

    def predict_next_position(self, x):
        return self.forward(x, batch_size=x[0].shape[0])

    def train_on_observations(self):
        entry = []
        data = Data(batch_size=3)
        for step in range(10000):
            X, Y = data.get_batch(mode='train')
            #print(X[0].shape)
            self.optimizer.zero_grad()
            self.hidden = (torch.zeros(1, 1, particles*2),
                           torch.zeros(1, 1, particles*2))

            train_loss = self.loss(self.predict_next_position(X), Y)
            train_loss.backward()
            self.optimizer.step()

            print(f'step {step}: {train_loss.item()}')
            entry.append({'time_step': step, 'loss': train_loss.item(), 'type': 'train'})

            if step % 10 == 0:
                torch.save(model.state_dict(), saved_path)


        sns.lineplot(data=pd.DataFrame(entry), x='time_step', y='loss', hue='type')
        pyplot.show()


model = Model()
model.train_on_observations()


def create_gif():
    """
    This function generates a gif to visualize the trajectory of the particles.
    :return:
    """
    import os
    import glob
    from PIL import Image

    fp_in = "/home/tadashi/Documents/plots/timestep_*.png"
    fp_out = "/home/tadashi/Documents/plots/dyari.gif"

    model = Model()
    model.load_state_dict(torch.load(saved_path))

    init_position = np.random.randn(2, particles) * 0.1
    position = np.reshape(init_position, (1, 1, particles * 2))

    traj_size = 100

    for time_step in range(0, traj_size):

        position = model.predict_next_position(x=position)
        pos = position.detach().numpy()
        pos = np.reshape(pos, (2, particles))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False, sharex=False)
        axes[0].set_title('Position in space')
        axes[1].set_title('Causal Graph')

        fig.suptitle(f'Timestep {time_step}')
        entries = []
        for particle_id in range(0, pos.shape[-1]):
            data = {'particle': particle_id,
                    'x coordinate': pos[0, particle_id],
                    'y coordinate': pos[1, particle_id]}
            entries.append(data)
        pdframe = pd.DataFrame(entries)

        pl = sns.scatterplot(data=pdframe, x='x coordinate', y='y coordinate', hue='particle', ax=axes[0])
        pl.set_ylim(-1.0 * 5, 5)
        pl.set_xlim(-1.0 * 5, 5)
        _path = '/home/tadashi/Documents/plots'
        plt.savefig(f"{_path}/timestep_{time_step}.png")
        plt.clf()

    # ref: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    _path = '/home/tadashi/Documents/plots'
    img, *imgs = [Image.open(f"{_path}/timestep_{i}.png") for i in range(0, traj_size)]
    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=10, loop=0)

    # delete all png files.
    for f in glob.glob(fp_in):
        os.remove(f)

def generate_trajectory():
    model = Model()
    model.load_state_dict(torch.load(saved_path))

    init_position = np.random.randn(2, particles) * 0.1
    position = np.reshape(init_position, (1, 1, particles * 2))
    traj_size = 100
    for i in range(0, traj_size):
        position = model.predict_next_position(x=position)
        pos = position.detach().numpy()
        pos = np.reshape(pos, (2, particles))
        entries = []
        for particle_id in range(0, pos.shape[-1]):
            data = {'particle': particle_id,
                    'x coordinate': pos[0, particle_id],
                    'y coordinate': pos[1, particle_id]}
            entries.append(data)
            pdframe = pd.DataFrame(entries)

    print(pdframe)

create_gif()
#generate_trajectory()


