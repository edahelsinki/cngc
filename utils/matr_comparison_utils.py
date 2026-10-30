import numpy as np
import torch

from dataset.data_utils import get_data_loaders
from utils.estimators import get_gradient, get_hessian


def compare_hessian_eigv(model, train_dataloader, batch_size, class_task=False,
                         subtract_clean=False, binary_noise=False, noise_mult=1.0,
                         verbose=True):
    gradients_noisy, gradients_clean, hessians_clean = [], [], []
    for x_batch, y_batch, _ in train_dataloader:
        model.zero_grad()
        # get noisy targets
        if binary_noise:
            noise_unit = torch.randint(0, 2, size=y_batch.shape).float() * 2 - 1
        else:
            noise_unit = torch.randn(size=y_batch.shape)

        if class_task:
            pred = model(x_batch)
            v = torch.sqrt(torch.sigmoid(pred) * (1. - torch.sigmoid(pred))).detach()
            noise = np.sqrt(batch_size) * noise_unit * v
        else:
            noise = np.sqrt(batch_size) * noise_unit

        y_batch_noisy = y_batch + noise * noise_mult
        grad_noisy = get_gradient(model, x_batch, y_batch_noisy.detach(), class_task=class_task, randomness="different")
        if subtract_clean:
            model.zero_grad()
            grad_clean = get_gradient(model, x_batch, y_batch.detach(), class_task=class_task, randomness="different")
            grad_noisy = grad_noisy - grad_clean
        gradients_noisy.append(grad_noisy)

        model.zero_grad()
        hessian_clean = get_hessian(model, x_batch, y_batch.detach(), class_task=class_task, randomness="different")
        hessians_clean.append(hessian_clean)

    gradients_noisy = torch.stack(gradients_noisy, dim=0).T
    grad_cov_noisy = torch.cov(gradients_noisy) / (noise_mult ** 2)
    grad_cov_noisy = grad_cov_noisy.float()

    hessian = torch.mean(torch.stack(hessians_clean, dim=0), dim=0).detach()

    grad_cov_noisy_eigv = torch.linalg.eigvalsh(grad_cov_noisy)
    hess_eigv = torch.linalg.eigvalsh(hessian)
    hess_eigv_relu = torch.relu(hess_eigv)

    if verbose:
        print(f"Hessian eigv:")
        print([f"{x:.5f}" for x in sorted(hess_eigv.numpy())])
        print()
        print(f"Noisy grad cov eigv:")
        print([f"{x:.5f}" for x in sorted(grad_cov_noisy_eigv.numpy())])
        print()

    grad_cov_hess_eigv_diff_rel = torch.linalg.norm(grad_cov_noisy_eigv - hess_eigv_relu, ord=2) / torch.linalg.norm(hess_eigv_relu, ord=2)
    print(f"Eigv diff rel (noisy cov vs Hessian) = {grad_cov_hess_eigv_diff_rel}")

    return grad_cov_hess_eigv_diff_rel


def test_noise(model_dist, X, y, batch_size=128, class_task=False, verbose=True):
    train_dataloader, _ = get_data_loaders(X, y[:, None], batch_size=batch_size, train_idx=len(X))

    print("Gaussian")
    eigv_diff_gauss = compare_hessian_eigv(model_dist, train_dataloader, batch_size=batch_size,
                                           class_task=class_task, subtract_clean=False,
                                           binary_noise=False, verbose=verbose)

    print("Binary")
    eigv_diff_bin = compare_hessian_eigv(model_dist, train_dataloader, batch_size=batch_size,
                                         class_task=class_task, subtract_clean=False,
                                         binary_noise=True, verbose=verbose)

    print("Binary, clean subtracted")
    eigv_diff_bin_cl = compare_hessian_eigv(model_dist, train_dataloader, batch_size=batch_size,
                                            class_task=class_task, subtract_clean=True,
                                            binary_noise=True, verbose=verbose)

    print("Binary, mult = 10")
    eigv_diff_bin_mult = compare_hessian_eigv(model_dist, train_dataloader, batch_size=batch_size,
                                              class_task=class_task, subtract_clean=False,
                                              noise_mult=10., binary_noise=True,
                                              verbose=verbose)

    return eigv_diff_gauss, eigv_diff_bin, eigv_diff_bin_cl, eigv_diff_bin_mult


def get_matr_diff(gt_matr, matr):
    diff = torch.linalg.norm(gt_matr - matr, ord=2) / (torch.linalg.norm(gt_matr, ord=2) + 1e-8)
    return diff
