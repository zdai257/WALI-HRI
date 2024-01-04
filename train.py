import os
import yaml
import torch
import numpy as np
from torch.optim import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.data_loader import build_data_loader
from models.lstm import build_lstm


def main():
    lr = 0.001
    batch_size = 64
    num_epochs = 100
    early_stop_patience = 5

    model, criteria = build_lstm()

    optimizer = RMSprop(model.parameters(), lr=lr)

    train_loader = build_data_loader()
    val_loader = build_data_loader()
    test_loader = build_data_loader()

    best_val_loss = float('inf')
    early_stop_count = 0

    for epoch in range(num_epochs):

        # Shape of inputs: (batch_size, sequence_length, num_features)
        # Shape of targets: (batch_size, num_features)
        for idx, (inputs, targets) in enumerate(train_loader):

            pred = model(inputs)

            loss = criteria(pred, targets)

            # Backward pass and optimization (for training)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO: eval with validation loss and other metrics
            val_loss = 0

            # Example early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_count = 0
                # Save the best model if needed
                # torch.save(model.state_dict(), 'best_model.pt')
            else:
                early_stop_count += 1
                if early_stop_count >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch}. Best validation loss: {best_val_loss}")
                    break


if __name__ == "__main__":
    main()

