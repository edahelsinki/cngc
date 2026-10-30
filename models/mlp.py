import torch
import torch.nn as nn
from models.base_model import ModuleWithDistortion


class MLPRegressor(ModuleWithDistortion):
    def __init__(self, in_features, out_features, hidden_layers=[10], add_bias=False, is_relu=True):
        super(MLPRegressor, self).__init__()
        layers = []
        prev_layer_size = in_features
        for layer_size in hidden_layers:
            layers.append(nn.Linear(prev_layer_size, layer_size, bias=add_bias))
            activation = nn.ReLU() if is_relu else nn.Tanh()
            layers.append(activation)
            prev_layer_size = layer_size
        layers.append(nn.Linear(prev_layer_size, out_features, bias=add_bias))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_weights(self):
        """
        Get the weights and bias of the MLP.
        :return: Weights and bias of the MLP.
        """
        weights = []
        for name, param in self.named_parameters():
            weights.append(param.detach().clone().flatten())
        return torch.cat(weights)
