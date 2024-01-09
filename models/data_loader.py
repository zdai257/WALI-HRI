import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import pickle
from math import pi


class WALIHRIDataset(Dataset):
    def __init__(self, data,
                 samp_list,  # specify sample Key other than current split
                 tau=5, freq=0.2):
        self.data = data
        self.sequence_length = int(tau / freq) + 1

        self.data_mean, self.data_sd = self.get_mean_sd()

        # discuss data by sample_id ('GBSIOT_XX') and sampling idx
        train_sequences = []
        train_y = []

        for index, (key, val) in enumerate(self.data.items()):
            if key not in samp_list:
                continue

            else:

                # col No. 16 is "annotation", No. 0 is "timestamp"
                #feats_col = val.iloc[:, np.r_[2:16, 17:]]

                feats_cols = val.drop(columns=['annotation', 'timestamp'])
                ### 1. Normalise ###
                feats_cols = self.normalise(feats_cols)

                ### 2. fill NaN with 0s ###
                feats_cols.fillna(0, inplace=True)

                feats = feats_cols.values  # .to_numpy()

                ### 3. Sequence Zero-padding ###
                # select only sequences with a non-NaN y label, and allow self.sequence_length without padding
                for i in range(len(feats) - self.sequence_length):

                    labels = val.iloc[i:i + self.sequence_length, val.columns.get_loc('annotation')]
                    labels.fillna(0, inplace=True)
                    # skip samples with 'NaN' annotation
                    if labels.isnull().values.any():
                        continue

                    # get only the last timestamp's annotation as Label
                    train_y.append(labels[-1])

                    sequence = feats[i:i + self.sequence_length]
                    train_sequences.append(sequence)

                    #train_y.append(val.iloc[self.sequence_length:, val.columns.get_loc('annotation')])

        assert len(train_sequences) == len(train_y)

        # Shape of X: (num_sequences, sequence_length, num_features)
        self.X = np.array(train_sequences)
        # Shape of Y: (num_sequences, sequence_length, labels)
        self.Y = np.expand_dims(np.array(train_y), axis=1)

    def normalise(self, df):
        df = df - self.data_mean
        return df / (self.data_sd + 1e-12)

    def get_mean_sd(self):
        whole_data = pd.concat(list(self.data.values()), axis=0)
        whole_mean = whole_data.mean()
        whole_sd = whole_data.std()
        whole_max = whole_data.max()
        whole_min = whole_data.min()
        #print(whole_mean, whole_sd)
        # Manually set Nomalise scale
        whole_mean['rotation_x'] = 0.
        whole_mean['rotation_y'] = 0.
        whole_mean['rotation_z'] = 0.
        whole_sd['rotation_x'] = pi/2
        whole_sd['rotation_y'] = pi/2
        whole_sd['rotation_z'] = pi/2
        whole_mean['pitch'] = 0.
        whole_mean['yaw'] = 0.
        whole_mean['roll'] = 0.
        whole_sd['pitch'] = 90.
        whole_sd['yaw'] = 90.
        whole_sd['roll'] = 90.
        whole_mean['norm_pos_x'] = 0.5
        whole_mean['norm_pos_y'] = 0.5
        whole_mean['fixa_conf'] = 0.5
        whole_mean['blink_conf'] = 0.5
        whole_mean['in_surface'] = 0.5
        whole_sd['norm_pos_x'] = 0.25
        whole_sd['norm_pos_y'] = 0.25
        whole_sd['fixa_conf'] = 0.5
        whole_sd['blink_conf'] = 0.5
        whole_sd['in_surface'] = 0.5
        # Mels
        #print(whole_mean.index)
        for col in list(whole_mean.index):
            if col.startswith('mel'):
                whole_mean[col] = -80.
                whole_sd[col] = 40.

        return whole_mean.drop(labels=['annotation', 'timestamp']), whole_sd.drop(labels=['annotation', 'timestamp'])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y


def build_data_loader(config, ctype=None):
    exported_pkl = config['data']['pkl_name']

    with open(exported_pkl, 'rb') as handle:
        my_data = pickle.load(handle)

    if ctype is None:
        else_lst = list(my_data.keys())
        for x in list(my_data.keys()):
            for sample in config['data']['val_samples'] + config['data']['test_samples']:
                if sample in x:
                    else_lst.remove(x)
                    break

        train_dataset = WALIHRIDataset(my_data, else_lst, tau=config['model']['seq_length_s'])
    elif ctype == 'val':
        else_lst = []
        for x in list(my_data.keys()):
            for sample in config['data']['val_samples']:
                if sample in x:
                    else_lst.append(x)
                    break

        train_dataset = WALIHRIDataset(my_data, else_lst, tau=config['model']['seq_length_s'])
    elif ctype == 'test':
        else_lst = []
        for x in list(my_data.keys()):
            for sample in config['data']['test_samples']:
                if sample in x:
                    else_lst.append(x)
                    break

        train_dataset = WALIHRIDataset(my_data, else_lst, tau=config['model']['seq_length_s'])
    else:
        raise TypeError("Illegal split")


    #print(train_dataset.X, train_dataset.Y)
    print("\nSplit shape: ")
    print(train_dataset.X.shape, train_dataset.Y.shape)

    # Allow class-balancing in sampling with WeightedRandomSampler
    class_labels = train_dataset.Y[:, -1]
    # t should be 0 / 1
    class_sample_count = np.array([len(np.where(class_labels == t)[0]) for t in np.unique(class_labels)])
    #print(class_labels, class_sample_count)

    class_weights = 1.0 / class_sample_count
    weights = class_weights[list(int(x) for x in class_labels)]
    print(weights, class_weights)

    # Create a WeightedRandomSampler to ensure balanced class distribution
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    #print(list(sampler)[:100])

    # Define batch size and create a DataLoader using the sampler
    if ctype == 'test':
        batch_size = 1
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=None)
    else:
        batch_size = config['training']['batch_size']
        data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    return data_loader
