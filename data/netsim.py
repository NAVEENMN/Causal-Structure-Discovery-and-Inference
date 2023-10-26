#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Netsim data can be downloaded from https://www.fmrib.ox.ac.uk/datasets/netsim/.
This script converts data to a compatible format to run across all causal methods.
"""
import os
import json
from Utils import Log
from experiment import Observations
from experiment import Experiment


class NetSim(Experiment):
    def __init__(self, experiment_id):
        super().__init__(experiment_id)
        self.num_channels = 0
        self.num_subjects = 0
        self.length = 0

    def get_num_of_channels(self):
        return self.num_channels

    def get_num_of_subjects(self):
        return self.num_subjects

    def set_num_of_channels(self, value):
        self.num_channels = value

    def set_num_of_subjects(self, value):
        self.num_subjects = value

    def set_length(self, length):
        self.length = length

    def get_length(self):
        return self.length

    def _get_channel_vars(self):
        _nc = self.get_num_of_channels()
        column_names = []
        column_names.extend([f'channel_{_id}' for _id in range(_nc)])
        return column_names

    def _get_edge_vars(self):
        np = self.get_num_of_channels()
        column_names = []
        for i in range(np):
            for j in range(np):
                if i != j:
                    column_names.append(f'e_{i}_{j}')
        return column_names

    def get_channel_observational_record(self):
        channel_observations = Observations()
        _vars = self._get_channel_vars()
        _vars.append('step')
        _vars.append('subject')
        channel_observations.set_column_names(columns=_vars)
        return channel_observations

    def get_edge_observational_record(self):
        edge_observations = Observations()
        _vars = self._get_edge_vars()
        _vars.append('step')
        _vars.append('subject')
        edge_observations.set_column_names(columns=_vars)
        return edge_observations

    def save(self):
        Log.info("Netsim", "Settings", f"Saving settings for experiment {self._id}")
        exp_data = dict()
        if os.path.exists(self.experiment_path):
            with open(self.experiment_path) as json_file:
                exp_data = json.load(json_file)
        exp_data[self._id]['settings']['netsim']['length'] = self.get_length()
        exp_data[self._id]['settings']['netsim']['num_channels'] = self.get_num_of_channels()
        exp_data[self._id]['settings']['netsim']['num_subjects'] = self.get_num_of_subjects()
        exp_data[self._id]['settings']['netsim']['channel_variables'] = self._get_channel_vars()
        exp_data[self._id]['settings']['netsim']['edges_variables'] = self._get_edge_vars()
        with open(self.experiment_path, 'w') as f:
            json.dump(exp_data, f, indent=4)
        Log.info('Netsim', 'Settings', f"Saved experiment {self._id} settings to {self.experiment_path}")
