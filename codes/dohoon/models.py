from torch import nn
from torch.nn import functional as F


class FeedForwardNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout_p):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        self.layers.append(nn.Dropout(dropout_p[0]))
        self.layers.append(nn.ReLU())

        layer_sizes = tuple(zip(hidden_layers[:-1], hidden_layers[1:]))
        for i in range(len(layer_sizes)):
            self.layers.append(nn.Linear(*layer_sizes[i]))
            self.layers.append(nn.Dropout(dropout_p[i + 1]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_layers[-1], output_size))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class FeedForwardNetOne(nn.Module):
    def __init__(self, n_neurons=16, dropout_rate=0):
        super().__init__()
        self.n_neurons = n_neurons

        self.lin1 = nn.Linear(1024, n_neurons)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lin2 = nn.Linear(n_neurons, 5)

    def forward(self, x):
        out = self.act1(self.lin1(x))
        out = self.dropout1(out)
        out = self.lin2(out)
        return out


class FeedForwardNetTwo(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(1024, 128)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(128, 16)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(16, 5)

    def forward(self, x):
        out = self.act1(self.lin1(x))
        out = self.act2(self.lin2(out))
        out = self.lin3(out)
        return out


class FeedForwardNetTwoDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(1024, 64)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.lin2 = nn.Linear(64, 16)
        self.dropout2 = nn.Dropout(0.5)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(16, 5)

    def forward(self, x):
        out = self.act1(self.lin1(x))
        out = self.dropout1(out)
        out = self.act2(self.lin2(out))
        out = self.dropout2(out)
        out = self.lin3(out)
        return out
