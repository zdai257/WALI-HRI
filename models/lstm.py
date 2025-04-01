import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


class EALSTM(nn.Module):
    def __init__(self, n_features=54, n_hidden=32, n_layers=2, n_output=1, weight=None, bidirectional=True):
        super(EALSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.weight = make_cuda(torch.FloatTensor(weight))
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden,
                            num_layers=n_layers, bidirectional=bidirectional,
                            batch_first=True)
        self.regr = nn.Linear(2 * n_hidden if bidirectional else n_hidden, n_output)

    def forward(self, inputs):  # inputs: [batch_size, time_steps, n_features]
        n_batch = inputs.size(0)
        inputs = torch.transpose(torch.transpose(inputs, -1, -2) * self.weight, -1, -2)
        _, (hidden, cell) = self.lstm(inputs)
        hidden = hidden.view(self.n_layers, 2 if self.bidirectional else 1, n_batch, self.n_hidden)
        if self.bidirectional:
            f_hidden, b_hidden = hidden[-1]
            hidden = torch.cat((f_hidden, b_hidden), dim=1)
        else:
            hidden = hidden[-1]
        out = self.regr(hidden)
        return out.view(-1)


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

    elif config['model']['name'] == "EALSTM":
        model = EALSTM(n_features=config['model']['input_dim'],
                       n_hidden=config['model']['hidden_dim1'],
                       weight=21 * [1.0],
                       bidirectional=True
                       )

    else:
        raise NameError("Does not support this model name!")
    #print(model)

    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification

    return model, criterion
