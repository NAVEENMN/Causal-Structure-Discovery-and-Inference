import torch
import random
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns

time_slice = 4
observation_length = 4
particles = 5
features = 5
trajectory_length = 1000
sample_freq = 10


class Data:
    def __init__(self, batch_size=2, time_lag=4):
        self.data = pd.read_pickle('dyari.pkl')
        self.train = self.data.sample(frac=0.8, random_state=200)
        self.test = self.data.drop(self.train.index)
        self.batch_size = batch_size
        self.time_lag = time_lag
        self.trajectory_length = trajectory_length

    def get_batch(self, mode='train'):
        x, y = [], []

        batch = self.train if mode == 'train' else self.test

        # Sample a random trajectories
        _x, _y = [], []
        trajectory_ids = random.sample(range(0, len(batch)), self.batch_size)
        observations = []
        for _id in trajectory_ids:
            positions = self.train.iloc[_id].trajectories.positions
            # random observation point
            index = random.randint(0, (len(positions)-(observation_length+2)))
            # make an observation
            observation = positions.iloc[index:index+observation_length+1]
            observations.append(observation)

        _reshape = lambda _x: np.reshape(_x.to_numpy(), (1, particles * 2))

        batching = dict()
        for i, obs in enumerate(observations):
            #print(f'observation {i}')
            for time_stamp in range(observation_length+1):
                #print(obs.iloc[time_stamp])
                if time_stamp not in batching:
                    batching[time_stamp] = []
                batching[time_stamp].append(_reshape(obs.iloc[time_stamp]))

        X = [np.reshape(np.asarray(batching[time_stamp]), (-1, particles*2)) for time_stamp in range(observation_length)]
        Y = [np.reshape(np.asarray(batching[observation_length]), (-1, particles*2))]
        return X, Y


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input dim is 8, output dim is 8
        self.lstm = nn.LSTM(particles*2, particles*2)
        # initialize the hidden state.
        self.hidden = (torch.randn(1, 1, particles*2), torch.randn(1, 1, particles*2))

    def forward(self, x):
        x = torch.Tensor(x)
        for i in x:
            # Step through the sequence one element at a time.
            # after each step, hidden contains the hidden state.
            out, self.hidden = self.lstm(i.view(3, 1, -1), self.hidden)
        return out


class Model(Net):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.parameters(),
                                         lr=0.001,
                                         momentum=0.9)

    def loss(self, y_prediction, y_real):
        y_real = torch.Tensor(y_real)
        loss = self.criterion(y_prediction, y_real)
        return loss

    def print_params(self, x):
        for param in self.parameters():
            print(param)

    def predict_next_position(self, x):
        return self.forward(x)

    def train(self):
        entry = []
        batch_size = 3
        data = Data(batch_size=batch_size)
        for step in range(100):
            X, Y = data.get_batch()
            #print(X[0].shape)
            self.optimizer.zero_grad()
            self.hidden = (torch.zeros(1, 1, particles*2),
                           torch.zeros(1, 1, particles*2))

            y_pred = self.predict_next_position(X)

            train_loss = self.loss(self.predict_next_position(X), Y)
            train_loss.backward()
            self.optimizer.step()

            print(f'step {step}: {train_loss.item()}')
            entry.append({'time_step': step, 'loss': train_loss.item(), 'type': 'train'})

        sns.lineplot(data=pd.DataFrame(entry), x='time_step', y='loss', hue='type')


model = Model()
model.train()