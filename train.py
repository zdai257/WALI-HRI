import os
import yaml
import torch
import numpy as np
from torch.optim import RMSprop, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tqdm
from models.data_loader import build_data_loader
from models.lstm import build_lstm


def main():
    # Load config file
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # load config
    lr = config['optimizer']['lr']
    num_epochs = config['training']['epochs']
    early_stop_patience = config['training']['early_stop_patience']

    model, criteria = build_lstm(config)

    if config['optimizer']['name'] == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=lr)
    elif config['optimizer']['name'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif config['optimizer']['name'] == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=lr)
    else:
        raise TypeError("Optimizer not supported!")

    # Define a linear learning rate scheduler
    lr_step_size = config['optimizer']['lr_step_size']
    gamma = config['optimizer']['gamma']  # Factor by which the learning rate will be reduced
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

    train_loader = build_data_loader(config)
    val_loader = build_data_loader(config, 'val')
    #test_loader = build_data_loader(config, 'test')

    best_val_loss = float('inf')
    early_stop_count = 0

    for epoch in range(num_epochs):
        print("\nEpoch No. {}:".format(epoch))

        epoch_loss = 0.0
        total = len(train_loader)

        with tqdm.tqdm(total=total) as pbar:
            # Shape of inputs: (batch_size, sequence_length, num_features)
            # Shape of targets: (batch_size, sequence_length, binary_label)
            for idx, (inputs, targets) in enumerate(train_loader):
                model.train()
                criteria.train()

                pred = model(inputs)

                loss = criteria(pred, targets)
                # print("Out shape: ", pred.shape)
                loss_value = loss.item()
                epoch_loss += loss_value

                # Backward pass and optimization (for training)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)

        print("Training loss: ", epoch_loss / total)

        ### Eval ###
        model.eval()
        criteria.eval()

        val_loss = 0.0
        val_total = len(val_loader)

        with tqdm.tqdm(total=val_total) as pbar:
            with torch.no_grad():
                for idx, (inputs, targets) in enumerate(val_loader):

                    val_pred = model(inputs)

                    loss = criteria(val_pred, targets)
                    val_loss += loss.item()
                    pbar.update(1)

            print("Validation loss: ", val_loss / val_total)

        # Example early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss / val_total
            early_stop_count = 0
            # Save the best model if needed
            if not os.path.exists(config['training']['saved_dir']):
                os.makedirs(config['training']['saved_dir'])
            torch.save(model, os.path.join(config['training']['saved_dir'], config['training']['saved_best_model']))
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best validation loss: {best_val_loss}")
                break

        scheduler.step()


if __name__ == "__main__":
    main()
