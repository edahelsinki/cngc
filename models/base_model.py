import torch
import torch.nn as nn


class ModuleWithDistortion(nn.Module):
    def get_param_distortion(self, sensitivities=None):
        """
        Get distortion vectors scaled by inverse sensitivity so that each
        parameter causes equal MSE impact per unit of distortion norm.

        :param sensitivities: dict from get_param_sensitivity(). If None,
                              use a random vector.
        :return: dict of distortion vectors, one per parameter.
        """
        distortions = {}
        for name, param in self.named_parameters():
            rand_vector = torch.randn_like(param)
            norm = torch.linalg.norm(rand_vector)
            unit_distortion = rand_vector / norm

            if sensitivities is not None:
                s = sensitivities.get(name, 1.0)
                # Scale inversely: insensitive params get larger geometric distortion
                # so their MSE contribution matches sensitive ones
                unit_distortion = unit_distortion / (s + 1e-8)
                # Re-normalise so dist_norm_w still controls the total budget
                unit_distortion = unit_distortion / torch.linalg.norm(unit_distortion)

            distortions[name] = unit_distortion
        return distortions

    def set_params(self, distortions, distortion_norm_w=0.1, distortion_norm_b=0.1):
        """
        Set the parameters of the network by adding to each parameter a distortion vector multiplied by a distortion norm.
        :param distortions: A dictionary with the distortion vectors for each model parameter.
        :param distortion_norm_w: The norm of the distortion to apply to the weights.
        :param distortion_norm_b: The norm of the distortion to apply to the biases.
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                v = distortions[name]
                if 'weight' in name:
                    # Apply a distortion to the weights
                    param.add_(v * distortion_norm_w)
                elif 'bias' in name:
                    # Apply a small distortion to the bias
                    param.add_(v * distortion_norm_b)

    def get_n_params(self):
        """
        Get the number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters())
