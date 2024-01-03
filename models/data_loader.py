import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import pickle


class WALIHRIDataset(Dataset):
    def __init__(self, data, tau=3, freq=0.2):
        self.data = data
        # TODO: discuss data by sample_id ('GBSIOT_XX') and sampling idx

        self.sequence_length = int(tau / freq) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.sequence_length, 1:]  # Assuming features start from column index 1
        y = self.data[idx+self.sequence_length, 1:]     # Assuming features start from column index 1
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


if __name__=="__main__":
    exported_pkl = './data_pkl.pkl'

    with open(exported_pkl, 'rb') as handle:
        my_data = pickle.load(handle)

    dataset = WALIHRIDataset(my_data)

    # Assuming you have a 'class_label' column in your DataFrame indicating classes
    # You can calculate class weights for WeightedRandomSampler
    class_labels = my_data['class_label']  # Replace 'class_label' with your actual class column name
    class_sample_count = np.array([len(np.where(class_labels == t)[0]) for t in np.unique(class_labels)])
    class_weights = 1.0 / class_sample_count
    weights = class_weights[class_labels]

    # Create a WeightedRandomSampler to ensure balanced class distribution
    sampler = WeightedRandomSampler(weights, len(weights))

    # Define batch size and create a DataLoader using the sampler
    batch_size = 64
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Accessing the data using the DataLoader
    for idx, (inputs, targets) in enumerate(data_loader):
        # inputs and targets will contain batches of sequences and corresponding targets
        # Shape of inputs: (batch_size, sequence_length, num_features)
        # Shape of targets: (batch_size, num_features)
        pass
