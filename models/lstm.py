import torch
import torch.nn as nn


class LSTM1(nn.Module):
    def __init__(self, input_dim=54, hidden_dim1=32, hidden_dim2=2, output_dim=1, dropout_prob=0.1):
        super(LSTM1, self).__init__()
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim1)  # BatchNorm for layer1

        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_dim2, output_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim2)  # BatchNorm for layer1

        #self.fc2 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(1, x.size(0), self.hidden_dim1).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim1).to(x.device)
        h1 = torch.zeros(1, x.size(0), self.hidden_dim2).to(x.device)
        c1 = torch.zeros(1, x.size(0), self.hidden_dim2).to(x.device)

        out, _ = self.lstm1(x, (h0, c0))
        out = self.dropout1(out)
        out = self.relu(out)
        out = self.batchnorm1(out.permute(0, 2, 1)).permute(0, 2, 1)

        out, _ = self.lstm2(out, (h1, c1))
        out = self.dropout2(out)
        out = self.batchnorm2(out.permute(0, 2, 1)).permute(0, 2, 1)

        #out = self.fc1(out[:, -1, :])  # Taking the last output sequence for classification
        out = self.fc1(out)  # Taking (batch, seq_length, output_dim) as output

        #out = self.fc2(out)

        # Sigmoid activation is not needed with BCEWithLogitsLoss
        # TODO try output layer without FCs but Softmax activation
        #out = torch.sigmoid(out)
        return out


class GRU1(nn.Module):
    def __init__(self, input_dim=54, hidden_dim1=32, hidden_dim2=2, output_dim=1, dropout_prob=0.1):
        super(GRU1, self).__init__()
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.rnn1 = nn.GRU(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim1)  # BatchNorm for layer1

        self.rnn2 = nn.GRU(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_dim2, output_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim2)  # BatchNorm for layer1

        #self.fc2 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_dim1).to(x.device)
        h1 = torch.zeros(1, x.size(0), self.hidden_dim2).to(x.device)

        out, _ = self.rnn1(x, h0)
        out = self.dropout1(out)
        out = self.relu(out)
        out = self.batchnorm1(out.permute(0, 2, 1)).permute(0, 2, 1)

        out, _ = self.rnn2(out, h1)
        out = self.dropout2(out)
        out = self.batchnorm2(out.permute(0, 2, 1)).permute(0, 2, 1)

        #out = self.fc1(out[:, -1, :])  # Taking the last output sequence for classification
        out = self.fc1(out)  # Taking (batch, seq_length, output_dim) as output

        #out = self.fc2(out)

        # Sigmoid activation is not needed with BCEWithLogitsLoss
        # TODO try output layer without FCs but Softmax activation
        #out = torch.sigmoid(out)
        return out


def build_lstm(config):

    # Initialize the LSTM / GRU model
    if config['model']['name'] == "LSTM1":
        model = LSTM1(input_dim=config['model']['input_dim'],
                      hidden_dim1=config['model']['hidden_dim1'],
                      hidden_dim2=config['model']['hidden_dim2'],
                      #hidden_dim3=config['model']['hidden_dim3'],
                      dropout_prob=config['model']['dropout_prob']
                      )

    elif config['model']['name'] == "GRU1":
        model = GRU1(input_dim=config['model']['input_dim'],
                     hidden_dim1=config['model']['hidden_dim1'],
                     hidden_dim2=config['model']['hidden_dim2'],
                     #hidden_dim3=config['model']['hidden_dim3'],
                     dropout_prob=config['model']['dropout_prob']
                     )

    else:
        raise NameError("Does not support this model name!")
    #print(model)

    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification

    return model, criterion
