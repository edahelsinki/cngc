import torch
import torch.nn as nn
from models.base_model import ModuleWithDistortion


class LinearRegressor(ModuleWithDistortion):
    def __init__(self, in_features, out_features, add_bias=True):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=add_bias)
        self.add_bias = add_bias

    def forward(self, x):
        return self.linear(x)

    def set_params_wb(self, weights, bias):
        """
        Set the weights and bias of the linear layer.
        :param weights: Weights of the linear layer.
        :param bias: Bias of the linear layer.
        """
        if len(weights.shape) == 1:
            # If weights are a 1D tensor, reshape it to match the expected shape
            # print(f"Reshaping weights from {weights.shape} to (1, {weights.shape[0]})")
            weights = weights.reshape(1, -1)

        if isinstance(bias, float):
            # If bias is a scalar, reshape it to match the expected shape
            # print(f"Reshaping bias from scalar to (1)")
            bias = torch.tensor([bias], dtype=torch.float32)

        # Ensure weights and bias are tensors
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)
        if not isinstance(bias, torch.Tensor):
            bias = torch.tensor(bias, dtype=torch.float32)

        if weights.shape != self.linear.weight.shape:
            raise ValueError(f"Expected weights shape {self.linear.weight.shape}, "
                             f"but got {weights.shape}")

        if self.add_bias:
            if bias.shape != self.linear.bias.shape:
                raise ValueError(f"Expected bias shape {self.linear.bias.shape}, "
                                 f"but got {bias.shape}")

        # Copy the weights and bias to the model parameters
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'linear.weight' in name:
                    param.copy_(weights)
                elif 'linear.bias' in name:
                    param.copy_(bias)

    def get_weights(self):
        """
        Get the weights and bias of the linear layer.
        :return: Weights and bias of the linear layer.
        """
        weights = self.linear.weight.detach().clone()
        if self.add_bias:
            bias = self.linear.bias.detach().clone()
            current_weights = torch.hstack((weights, bias.reshape(-1, 1)))
        else:
            current_weights = weights
        return current_weights
