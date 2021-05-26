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
        if mode == 'train':
            trajectory_ids = random.sample(range(0, len(self.train)), self.batch_size)
        else:
            trajectory_ids = random.sample(range(0, len(self.test)), self.batch_size)

        # Sample a random observations
        for _id in trajectory_ids:
            positions = self.train.iloc[_id].trajectories.positions
            index = random.randint(0, len(positions)-observation_length)
            observations = positions.iloc[index:index+observation_length+1]
            _reshape = lambda _x: np.reshape(_x.to_numpy(), (1, particles*2))
            _x = [_reshape(observations.iloc[i]) for i in range(observation_length)]
            _y = [_reshape(observations.iloc[-1])]
            x.append(_x)
            y.append(_y)
        print(x)
        x = np.reshape(x, (self.batch_size, 1, particles*2))
        y = np.reshape(x, (self.batch_size, 1, particles*2))

        return x, y


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
            out, self.hidden = self.lstm(i.view(1, 1, -1), self.hidden)
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
        data = Data(batch_size=3)
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