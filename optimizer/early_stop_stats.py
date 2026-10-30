import torch
from optimizer.pc_optimizer import GradientMeanEstimator, Preconditioner


class CredibleValue:
    def __init__(self, param_sizes, n_batches, use_clean_grad=False, use_pc=True, noise_mult=1.0):
        self.n_batches = n_batches
        self.use_clean_grad = use_clean_grad
        self.noise_mult = noise_mult
        self.cr_mult = 1  # n_batches if use_clean_grad else 1
        self._grad_mean_calc = GradientMeanEstimator()

        if self.use_clean_grad:
            cov_pc_es = Preconditioner(param_sizes=param_sizes, inverse_method="pinv", noise_mult=1.)
        elif not use_pc:  # there are no preconditioner in the optimization, the matrix is calculated only for early stopping stats
            cov_pc_es = Preconditioner(param_sizes=param_sizes, inverse_method="pinv", noise_mult=self.noise_mult)
        else:
            cov_pc_es = None
        self._cov_pc_es = cov_pc_es

    def get_value(self, cov_pc=None):
        if cov_pc is None and self._cov_pc_es is None:
            raise ValueError("At least one of cov_pc or cov_pc_es must be provided")

        grad_mean = self._grad_mean_calc.finalise()
        if self._cov_pc_es is not None:
            self._cov_pc_es.finalise(grad=grad_mean)
            cov_pc = self._cov_pc_es

        p = grad_mean.shape[0]
        print(f"Number of parameters is ", p)

        grad_proj = cov_pc.apply(grad_mean)
        z = self.cr_mult * torch.dot(grad_mean, grad_proj)
        return z.item()

    def record_grad(self, clean_grad, noisy_grad):
        self._grad_mean_calc.update(clean_grad)
        if self._cov_pc_es is not None:
            if self.use_clean_grad:
                grad_es = clean_grad.clone().detach()
            else:
                grad_es = noisy_grad.clone().detach()
            self._cov_pc_es.update(grad_es)



