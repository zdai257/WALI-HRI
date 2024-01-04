import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import pickle


class WALIHRIDataset(Dataset):
    def __init__(self, data,
                 else_samp=['GBSIOT_07B', 'GBSIOT07D'],  # specify sample Key other than current split
                 tau=3, freq=0.2):
        self.data = data
        self.sequence_length = int(tau / freq) + 1

        # TODO: discuss data by sample_id ('GBSIOT_XX') and sampling idx
        train_sequences = []
        train_y = []
        for index, (key, val) in enumerate(self.data.items()):
            if key in else_samp:
                continue

            else:
                # col No. 16 is Annotation
                feats_col = val.iloc[:, np.r_[2:16, 17:]]
                feats = val[feats_col].values
                for i in range(len(feats) - self.sequence_length + 1):
                    sequence = feats[i:i + self.sequence_length + 1]
                    train_sequences.append(sequence)

                    train_y.append(val.iloc[self.sequence_length:, 16:17])

        assert len(train_sequences) == len(train_y)

        # TODO: deal with NaN data
        # Shape of X: (num_sequences, sequence_length, num_features)
        self.X = np.array(train_sequences)
        # Shape of Y: (num_sequences, num_features)
        self.Y = np.array(train_y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y


def build_data_loader():
    exported_pkl = './data_pkl.pkl'

    with open(exported_pkl, 'rb') as handle:
        my_data = pickle.load(handle)

    train_dataset = WALIHRIDataset(my_data)
    #val_dataset = WALIHRIDataset(my_data, else_samp=[])  # specify Excluded samples
    #test_dataset = WALIHRIDataset(my_data, else_samp=[])

    # Allow class-balancing in sampling with WeightedRandomSampler
    class_labels = train_dataset.Y[:, 0]
    # t should be 0 / 1
    class_sample_count = np.array([len(np.where(class_labels == t)[0]) for t in np.unique(class_labels)])
    class_weights = 1.0 / class_sample_count
    weights = class_weights[class_labels]

    # Create a WeightedRandomSampler to ensure balanced class distribution
    sampler = WeightedRandomSampler(weights, len(weights))

    # Define batch size and create a DataLoader using the sampler
    batch_size = 64
    data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    return data_loader
