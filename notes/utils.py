import os
import numpy as np
import random
import copy

from collections import defaultdict
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.func import jacfwd, functional_call, jacrev, vmap

import seaborn as sns
import matplotlib.pyplot as plt


# To ensure reproducibility on the cluster with np.linalg.solve (for the analytical solution)
# torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# switch off flash attention and memory efficient attention to avoid issues with backward pass with vmap and hessian calc
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def plot_cov_matrices(cov_matrix1, cov_matrix2, filename='covariance_heatmaps.png', use_cbar=True, is_ggn=False):
    """
    Plots two covariance matrices and their difference.

    Parameters:
    - cov_matrix1: First covariance matrix (2D array).
    - cov_matrix2: Second covariance matrix (2D array).
    """

    # Ensure the matrices are numpy arrays
    cov_matrix1 = np.asarray(cov_matrix1)
    cov_matrix2 = np.asarray(cov_matrix2)

    # Calculate the difference
    diff_matrix = np.abs(cov_matrix1 - cov_matrix2)

    vmin = min(cov_matrix1.min(), cov_matrix2.min())
    vmax = max(cov_matrix1.max(), cov_matrix2.max())

    # Create the plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Plot Matrix 1
    h1 = sns.heatmap(cov_matrix1, ax=axes[0], cmap='viridis', annot=True, fmt=".2f", cbar=use_cbar, vmin=vmin, vmax=vmax)
    h1_title = "GGN" if is_ggn else "Grad Covariance"
    axes[0].set_title(h1_title)

    # Plot Matrix 2
    h2 = sns.heatmap(cov_matrix2, ax=axes[1], cmap='viridis', annot=True, fmt=".2f", cbar=use_cbar, vmin=vmin, vmax=vmax)
    axes[1].set_title('Hessian')

    # Plot Difference Matrix
    sns.heatmap(diff_matrix, ax=axes[2], cmap='coolwarm', annot=True, fmt=".2f", cbar=use_cbar)
    diff_title = "Difference (GGN - Hessian)" if is_ggn else "Difference (Grad cov - Hessian)"
    axes[2].set_title(diff_title)

    plt.tight_layout()
    plt.savefig(f'vis/{filename}')
    plt.show()


def plot_cov_gnn_hess_matrices(cov_matrix, ggn, hessian, filename='cov_gnn_heatmaps.png', use_cbar=True):
    # Ensure the matrices are numpy arrays
    cov_matrix = np.asarray(cov_matrix)
    ggn = np.asarray(ggn)
    hessian = np.asarray(hessian)

    # Calculate the difference
    diff_cov = np.abs(cov_matrix - hessian)
    diff_ggn = np.abs(ggn - hessian)

    vmin = min(cov_matrix.min(), ggn.min(), hessian.min())
    vmax = max(cov_matrix.max(), ggn.max(), hessian.min())

    vmin_diff = min(diff_cov.min(), diff_ggn.min())
    vmax_diff = max(diff_cov.max(), diff_ggn.max())

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5))

    sns.heatmap(ggn, ax=axes[0], cmap='viridis', annot=True, fmt=".2f", cbar=use_cbar, vmin=vmin, vmax=vmax)
    axes[0].set_title('GGN')

    sns.heatmap(cov_matrix, ax=axes[1], cmap='viridis', annot=True, fmt=".2f", cbar=use_cbar, vmin=vmin, vmax=vmax)
    axes[1].set_title('Calibrated noisy grad cov')

    sns.heatmap(hessian, ax=axes[2], cmap='viridis', annot=True, fmt=".2f", cbar=use_cbar, vmin=vmin, vmax=vmax)
    axes[2].set_title('Hessian')

    sns.heatmap(diff_ggn, ax=axes[3], cmap='coolwarm', annot=True, fmt=".2f", cbar=use_cbar, vmin=vmin_diff, vmax=vmax_diff)
    axes[3].set_title('Difference (GGN - Hessian)')

    sns.heatmap(diff_cov, ax=axes[4], cmap='coolwarm', annot=True, fmt=".2f", cbar=use_cbar, vmin=vmin_diff, vmax=vmax_diff)
    axes[4].set_title('Difference (Noisy grad cov - Hessian)')

    plt.tight_layout()
    plt.savefig(f'vis/{filename}')
    plt.show()


def generate_data_nn(model, n_samples=100000, n_features=10, scale=1, noise_std=0.1, class_task=False, multivar=False):
    np.random.seed(42)
    if multivar:
        X = gen_multivar_data(n_samples, n_features, scale)
    else:
        X = np.random.normal(loc=0, scale=scale, size=(n_samples, n_features))

    noise = np.random.normal(loc=0, scale=noise_std, size=(n_samples, 1))
    X_tensor = torch.tensor(X).float()
    y_true = model(X_tensor).detach().numpy()
    y = y_true + noise
    y = y.reshape(-1)  # Ensure y is a 1D array

    if class_task:
        prob = 1. / (1. + np.exp(-y))
        y = np.random.binomial(1, prob)
    return X, y


def gen_multivar_data(n_samples=100000, n_features=10, scale=1):
    dim = n_features
    rand_matr = np.random.rand(dim, dim)
    scale_vector = np.random.randint(1, scale + 1, dim)
    sigma_matr = rand_matr @ rand_matr.T + scale_vector ** 2 * np.identity(dim)  # make it positive definite
    X_multi = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma_matr, size=n_samples)
    return X_multi


def distort_model(model_gen, dist_norm_w, X=None, y=None):
    # print the number of model parameters:
    total_params = sum(p.numel() for p in model_gen.parameters())
    print(f"Total number of model parameters: {total_params}")

    # distort the model weights
    model = copy.deepcopy(model_gen)
    model_dist = copy.deepcopy(model_gen)
    w_distortions = model.get_param_distortion()
    model_dist.set_params(w_distortions, distortion_norm_w=dist_norm_w, distortion_norm_b=0.1)

    if X is not None and y is not None:
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y[:, None]).float()
        pred = model_dist(X_tensor)
        mse_dist = torch.mean((pred - y_tensor)**2).item()
        mse_gt = torch.mean((model_gen(X_tensor) - y_tensor)**2).item()
        print(f"MSE with GT model: {mse_gt:.4f}")
        print(f"MSE with distorted model: {mse_dist:.4f}")
    return model_dist


def get_data_loaders(X, y, batch_size=32, test_ratio=0.2, norm_X=False, norm_y=False, return_ystd=False, noise=None,
                     class_task=False, verbose=False):
    """
    Splits the dataset into training and test sets and returns DataLoaders for both.

    Parameters:
    - X: Features (numpy array).
    - y: Labels (numpy array).
    - batch_size: Size of each batch.
    - train_ratio: Proportion of the dataset to include in the train split.

    Returns:
    - trainloader: DataLoader for the training set.
    - testloader: DataLoader for the test set.
    """
    # split X and y to minibatches
    train_idx = int(len(X) * (1 - test_ratio)) // batch_size * batch_size

    if verbose:
        print("Train index is ", train_idx, len(X))

        print(f"Number of train points is {train_idx}, \n"
              f"number of batches is {train_idx // batch_size}, \n"
              f"number of test points is {len(X) - train_idx}")

    # split X and y to tensors
    X_train, y_train = to_tensor(X[:train_idx]), to_tensor(y[:train_idx])
    X_test, y_test = to_tensor(X[train_idx:]), to_tensor(y[train_idx:])

    if norm_X:
        X_mean = X_train.mean(dim=0)
        X_std = X_train.std(dim=0)
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
    else:
        # we need to subtract mean from X
        X_mean = X_train.mean(dim=0)
        X_train = X_train - X_mean
        X_test = X_test - X_mean

        if verbose:
            print("X_train std is ", X_train.std(dim=0))

    y_std = y_train.std(dim=0)
    if not class_task:
        if norm_y:
            y_mean = y_train.mean(dim=0)
            y_train = (y_train - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std
        else:
            # we need to subtract mean from y
            y_mean = y_train.mean(dim=0)
            y_train = y_train - y_mean
            y_test = y_test - y_mean

            if verbose:
                print("y_train std is ", y_train.std(dim=0))

    if noise is None:
        y_train_noise = torch.zeros_like(y_train)
        y_test_noise = torch.zeros_like(y_test)
    else:
        y_train_noise = to_tensor(noise[:train_idx])
        y_test_noise = to_tensor(noise[train_idx:])

    trainset = TensorDataset(X_train, y_train, y_train_noise)
    trainloader = get_dataload_with_seed(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = TensorDataset(X_test, y_test, y_test_noise)
    testloader = get_dataload_with_seed(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    if return_ystd:
        return trainloader, testloader, y_std

    return trainloader, testloader


def get_dataload_with_seed(dataset, batch_size, shuffle, num_workers=0):
    """
    Returns a DataLoader with a fixed seed for multiple workers for reproducibility.
    :param dataset: Dataset to load.
    :param batch_size: Size of each batch.
    :param shuffle: Whether to shuffle the dataset.
    :param num_workers: Number of workers for the DataLoader.
    :return: DataLoader with a fixed seed for multiple workers.
    """
    if num_workers > 0:
        g = torch.Generator()
        g.manual_seed(0)

        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                worker_init_fn=seed_worker,
                                generator=g)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def get_gradient(model, inputs, outputs, return_names=False, layer_idx=None, class_task=False, randomness="different"):
    """
    Computes the gradient of the loss with respect to its parameters.
    :param model: The model to compute the gradient for.
    :param inputs: The input data.
    :return: batch gradient.
    """
    def fmodel(params, inputs):
        return functional_call(model, params, inputs)

    def floss(params, inputs, outputs):
        if class_task:
            # for classification task, we can use cross-entropy loss
            logits = fmodel(params, inputs)
            pred = torch.sigmoid(logits)
            return -torch.mean(outputs * torch.log(pred + 1e-8) + (1. - outputs) * torch.log(1. - pred + 1e-8))
        else:
            return 0.5 * torch.mean((fmodel(params, inputs) - outputs)**2)

    with torch.no_grad():
        # grad is a dictionary of keys (parameter names) and values (gradients)
        batch_size = inputs.shape[0]
        params = dict(model.named_parameters())
        if layer_idx is not None:
            layer_name = 'mlp.' + str(2 * layer_idx) + '.'
            params = {k: v for k, v in params.items() if k.startswith(layer_name)}

        grad = vmap(jacrev(floss), in_dims=(None, 0, 0), randomness=randomness)(params, inputs, outputs)
        keys = sorted(list(grad.keys()))
        grad_matrix = torch.cat([grad[key].view(batch_size, -1) for key in keys], dim=1)
        grad_mean = torch.mean(grad_matrix, dim=0)

    if return_names:
        return grad_mean, keys
    return grad_mean


def get_hessian(model, inputs, outputs, return_names=False, class_task=False, randomness="different"):
    """
    Compute the Hessian of 0.5 * mean((model(x)-y)**2)
    w.r.t. model parameters. Includes residual-dependent terms.
    Average in the loss function before computing the gradient.
    """
    def fmodel(params, x):
        return torch.func.functional_call(model, params, x)

    def floss(params, inputs, outputs):
        if class_task:
            # for classification task, we can use cross-entropy loss
            logits = fmodel(params, inputs)
            pred = torch.sigmoid(logits)
            return -torch.mean(outputs * torch.log(pred + 1e-8) + (1. - outputs) * torch.log(1. - pred + 1e-8))
        else:
            return 0.5 * torch.mean((fmodel(params, inputs) - outputs) ** 2)

    params = dict(model.named_parameters())
    # H = hessian(floss, argnums=(0))(params, inputs, outputs)  # it doesn't expose "randomness" argument
    inner_jac = jacrev(floss, argnums=0)
    H_func = jacfwd(inner_jac, argnums=0, randomness=randomness)
    H = H_func(params, inputs, outputs)

    # sort the keys()
    keys = sorted(params.keys())

    #flatten Hessian dict into blocks
    blocks = []
    for k1 in keys:
        row_blocks = []
        for k2 in keys:
            block = H[k1][k2]
            block = block.reshape(params[k1].numel(), params[k2].numel())
            row_blocks.append(block)
        blocks.append(torch.cat(row_blocks, dim=1))

    # assemble Hessian matrix
    H = torch.cat(blocks, dim=0)

    if return_names:
        return H, keys
    return H


def get_full_jac(model, inputs, class_task=False, randomness="different"):
    """
    Computes the gradient of the model's output with respect to its parameters.
    :param model: The model to compute the Jacobian for.
    :param inputs: The input data.
    :return: batch gradient.
    """
    def fmodel(params, inputs):
        return functional_call(model, params, inputs)

    # Jacobian wrt params for ONE sample
    def single_jac(params, x):
        return jacrev(fmodel)(params, x)

    def flatten_jacobian_mse(jac):
        B = next(iter(jac.values())).shape[0]
        mats = []
        for k in sorted(jac.keys()):
            mats.append(jac[k].reshape(B, -1))
        return torch.cat(mats, dim=-1)  # [B, P]

    def flatten_jacobian_ce(jac):
        B, C = next(iter(jac.values())).shape[:2]
        mats = []
        for k in sorted(jac.keys()):
            mats.append(jac[k].reshape(B, C, -1)) # [B, C, P]
        return torch.cat(mats, dim=-1)

    def flatten_jacobian(jac, class_task):
        if class_task:
            return flatten_jacobian_ce(jac)
        else:
            return flatten_jacobian_mse(jac)

    # Vectorize over batch
    params = dict(model.named_parameters())
    jac = vmap(single_jac, in_dims=(None, 0), randomness=randomness)(params, inputs)

    # flatten Jacobian
    jac_matrix = flatten_jacobian(jac, class_task) # B, P for MSE, B, C, P for CE (if C > 1)
    return jac_matrix


def get_full_ggn(model, inputs, class_task=False):
    jac = get_full_jac(model, inputs, class_task=class_task) # [B, P] for regression and [B, C, P] for classification, C = 1 (both for regression and for classification)
    if class_task:
        # for classification the GGN is J^T V J where V is the diagonal matrix with v_i = p_i * (1 - p_i)
        pred = model(inputs) # [B, C]
        v = (torch.sigmoid(pred) * (1. - torch.sigmoid(pred))).detach() # [B, C]
        v_mat = torch.diag_embed(v)  # [B, C, C]
        ggn = jac.transpose(1, 2) @ v_mat @ jac  # [B, P, P]
        ggn = torch.sum(ggn, dim=0)
    else:
        # for regression the GGN is J^T J
        # jac.T - [P, B] and jac is [B, P]
        ggn = jac.T @ jac  # [P, P]

    batch_size = inputs.shape[0]
    ggn  = ggn / batch_size
    return ggn


def get_sorted_hessian_block_mask(model, device="cpu"):
    """
    Builds a block-diagonal mask aligned with sorted named_parameters().
    Each module = one block.
    """
    params = dict(model.named_parameters())
    keys = sorted(params.keys())

    # we need to group weights and biases of the same module together, so we first create a mapping from parameter name to module name
    param_to_module = {}
    for module_name, module in model.named_modules():
        for pname, _ in module.named_parameters(recurse=False):
            full = f"{module_name}.{pname}" if module_name else pname
            param_to_module[full] = module_name or "root"

    # Assign flat indices + group per module (single pass)
    module_indices = defaultdict(list)
    start = 0

    for k in keys:
        p = params[k]
        n = p.numel()
        module = param_to_module[k]

        module_indices[module].extend(range(start, start + n))
        start += n

    P = start
    mask = torch.zeros(P, P, device=device)

    # Fill blocks
    for idx in module_indices.values():
        idx = torch.tensor(idx, device=device)
        mask[idx[:, None], idx[None, :]] = 1

    return mask
