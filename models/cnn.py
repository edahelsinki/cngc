import torch
import torch.nn as nn
from models.base_model import ModuleWithDistortion


class CNN1dRegressor(ModuleWithDistortion):
    def __init__(self, in_channels, out_channels, kernel_size, n_layers, seq_len, pool_size=1):
        super(CNN1dRegressor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.seq_len = seq_len

        conv_list = []
        pool_list = []
        for i in range(n_layers):
            conv = nn.Conv1d(in_channels, out_channels, kernel_size)
            conv_list.append(conv)
            in_channels = out_channels

            pool = nn.MaxPool1d(kernel_size=pool_size)
            pool_list.append(pool)

        self.conv_layers = nn.ModuleList(conv_list)
        self.pool_layers = nn.ModuleList(pool_list)

        lin_in_size = self.get_lin_input_size()
        self.linear = nn.Linear(lin_in_size, 1, bias=False)

    def get_lin_input_size(self):
        x = torch.ones(1, self.in_channels, self.seq_len)

        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = torch.relu(x)
            x = pool(x)
        x = x.flatten(start_dim=1)
        return x.shape[1]

    def forward(self, x):
        len_x = len(x.shape)
        no_batch = False
        if len(x.shape) < 3:
            if self.in_channels == 1:
                x = x.unsqueeze(-2)  # Add a channel dimension
            else:
                no_batch = True  # needed for vmap
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = torch.relu(x)
            x = pool(x)

        if len_x == 1 or no_batch:
            x = x.flatten()
        else:
            x = x.flatten(start_dim=1)

        x = self.linear(x)
        return x
