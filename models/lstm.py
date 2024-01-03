import torch
import torch.nn as nn
from torch.optim import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LSTM1(nn.Module):
    def __init__(self, input_dim=54, hidden_dim1=32, hidden_dim2=2, output_dim=1, dropout_prob=0.1):
        super(LSTM1, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out = self.relu(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc(out[:, -1, :])  # Taking the last output sequence for classification
        return out


def build_lstm():

    learning_rate = 0.001
    batch_size = 64
    num_epochs = 100
    early_stop_patience = 5

    # Initialize the LSTM model
    model = LSTM1()
    print(model); exit()

    optimizer = RMSprop(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification

    # Example data (replace this with your own dataset)
    # X, y = your_data  # Replace this with your actual data and labels
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Example training loop with early stopping
    best_val_loss = float('inf')
    early_stop_count = 0
    for epoch in range(num_epochs):
        # Training
        model.train()
        # Implement your training steps here using DataLoader or batches

        # Validation
        model.eval()
        # Implement your validation steps here using DataLoader or batches
        # Calculate validation loss and other metrics
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

