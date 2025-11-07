import numpy as np
import torch.nn as nn
import torch as trch


def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update"""
    
    Au = np.dot(A_inv, u)
    A_inv -= np.outer(Au, Au) / (1 + np.dot(u.T, Au))
    return A_inv


class Model(nn.Module):
    """Template for fully connected neural network for scalar approximation"""

    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers,
                 dropout_prob=0.0,
                 ):
        super(Model, self).__init__()

        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input layer
        self.lin1 = nn.Linear(input_size, hidden_size)
        w_a = np.random.normal(0, 4 / hidden_size, (int(input_size / 2), int(hidden_size / 2)))
        w_b = np.zeros((int(input_size / 2), int(hidden_size / 2)))
        w_1 = np.concatenate((w_a, w_b), 0)
        w_2 = np.concatenate((w_b, w_a), 0)
        w_layer = np.concatenate((w_1, w_2), 1)
        self.lin1.weight.data = trch.from_numpy(w_layer.T).float()

        # hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(self.n_layers)])
        for layer in self.hidden_layers:
            w_a = np.random.normal(0, 4 / hidden_size, (int(hidden_size / 2), int(hidden_size / 2)))
            w_b = np.zeros((int(hidden_size / 2), int(hidden_size / 2)))
            w_1 = np.concatenate((w_a, w_b), 0)
            w_2 = np.concatenate((w_b, w_a), 0)
            w_layer = np.concatenate((w_1, w_2), 1)
            layer.weight.data = trch.from_numpy(w_layer.T).float()

        # output layer
        self.lin5 = nn.Linear(hidden_size, 1)
        w_part1 = np.random.normal(0, 2 / hidden_size, int(hidden_size / 2))
        w_layer_out = np.concatenate((w_part1, -w_part1), 0).T
        self.lin5.weight.data = trch.from_numpy(w_layer_out)[None, :].float()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        a = self.lin1(x)
        for hl in self.hidden_layers:
            a = hl(nn.functional.relu(a))
        b = self.hidden_size ** 0.5 * self.lin5(nn.functional.relu(a))
        b = trch.sigmoid(b)
        return b
