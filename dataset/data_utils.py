import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from dataset.data_preprocessing import *


def get_data_loaders(X, y, batch_size=32, test_ratio=0.2, norm_X=False, norm_y=False, return_ystd=False, noise=None,
                     task='regr', train_idx=None, use_val=False, verbose=False):
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

    if train_idx is None:
        # split X and y to train and test sets
        if task == 'class':
            # (stratified to preserve class imbalance)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42,
                                                                stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
        train_idx = len(X_train)
    else:
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_test, y_test = X[train_idx:], y[train_idx:]

    if use_val:
        # further split train set to train and validation sets
        if task == 'class':
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_ratio, random_state=42,
                                                              stratify=y_train)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_ratio, random_state=42)
        train_idx = len(X_train)

    if verbose:
        print("Train index is ", train_idx, len(X))

        print(f"Number of train points is {train_idx}, \n"
              f"number of batches is {train_idx // batch_size}, \n"
              f"number of test points is {len(X) - train_idx}")

        if use_val:
            print(f"Number of validation points is {len(X_val)}")

    # split X and y to tensors
    X_train, y_train = to_tensor(X_train), to_tensor(y_train)
    X_test, y_test = to_tensor(X_test), to_tensor(y_test)

    if use_val:
        X_val, y_val = to_tensor(X_val), to_tensor(y_val)

    # select axes that are reduced for normalization (preserve feature axis=1)
    if X_train.ndim == 2:
        axes = 0
    else:
        axes = tuple(i for i in range(X_train.ndim) if i != 1)

    if norm_X:
        X_mean = X_train.mean(dim=axes, keepdim=True)
        X_std = X_train.std(dim=axes, keepdim=True)
        X_train = (X_train - X_mean) / (X_std + 1e-8)
        X_test = (X_test - X_mean) / (X_std + 1e-8)

        if use_val:
            X_val = (X_val - X_mean) / (X_std + 1e-8)
    else:
        # we need to subtract mean from X
        X_mean = X_train.mean(dim=axes, keepdim=True)
        X_train = X_train - X_mean
        X_test = X_test - X_mean

        if use_val:
            X_val = X_val - X_mean

    if verbose:
        print("X_train std is ", X_test.std(dim=axes, keepdim=True))

    y_std = y_train.std(dim=0)
    if task != 'class':
        if norm_y:
            y_mean = y_train.mean(dim=0)
            y_train = (y_train - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std
            if use_val:
                y_val = (y_val - y_mean) / y_std
        else:
            # we need to subtract mean from y
            y_mean = y_train.mean(dim=0)
            y_train = y_train - y_mean
            y_test = y_test - y_mean

            if use_val:
                y_val = y_val - y_mean

            if verbose:
                print("y_train std is ", y_train.std(dim=0))

    if noise is None:
        y_train_noise = torch.zeros_like(y_train)
        y_test_noise = torch.zeros_like(y_test)

        if use_val:
            y_val_noise = torch.zeros_like(y_val)
    else:
        if use_val:
            # the order of the noise assignment does not matter
            y_train_noise = to_tensor(noise[:train_idx])
            y_val_noise = to_tensor(noise[train_idx:train_idx + len(X_val)])
            y_test_noise = to_tensor(noise[train_idx + len(X_val):])
        else:
            y_train_noise = to_tensor(noise[:train_idx])
            y_test_noise = to_tensor(noise[train_idx:])

    trainset = TensorDataset(X_train, y_train, y_train_noise)
    trainloader = get_dataload_with_seed(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = TensorDataset(X_test, y_test, y_test_noise)
    testloader = get_dataload_with_seed(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    if use_val:
        valset = TensorDataset(X_val, y_val, y_val_noise)
        valloader = get_dataload_with_seed(valset, batch_size=batch_size, shuffle=False, num_workers=0)
        dataloaders = (trainloader, testloader, valloader)
    else:
        dataloaders = (trainloader, testloader)

    if return_ystd:
        return *dataloaders, y_std

    return dataloaders


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


def generate_data(n_samples=100000, n_features=10, scale=1, noise_std=0.1, w=None, b=None, class_task=False, multivar=False):
    np.random.seed(42)
    if multivar:
        X = gen_multivar_data(n_samples, n_features, scale)
    else:
        X = np.random.normal(loc=0, scale=scale, size=(n_samples, n_features))

    w = np.random.rand(n_features) if w is None else w
    b = 0.1 if b is None else b

    noise = np.random.normal(loc=0, scale=noise_std, size=(n_samples,))
    y_true = X @ w + b
    y = y_true + noise

    if class_task:
        prob = 1. / (1. + np.exp(-y))
        y = np.random.binomial(1, prob)

    return X, y, w, b


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


def generate_data_seq(n_samples=100000, n_features=10, scale=1, noise_std=0.1, w=None, b=None, class_task=False,
                      multivar=False, channels=3):
    np.random.seed(42)
    if multivar:
        X = gen_multivar_data(n_samples, n_features, scale)
    else:
        X = np.random.normal(loc=0, scale=scale, size=(n_samples, n_features))

    w = np.random.rand(n_features) if w is None else w
    b = 0.1 if b is None else b

    noise = np.random.normal(loc=0, scale=noise_std, size=(n_samples,))
    y_true = X @ w + b
    y = y_true + noise

    if class_task:
        prob = 1. / (1. + np.exp(-y))
        y = np.random.binomial(1, prob)

    X = np.expand_dims(X, axis=-2)
    X = np.tile(X, (1, channels, 1))
    return X, y


def generate_data_nn_seq(model, n_samples=100000, n_features=10, scale=1, noise_std=0.1, class_task=False,
                         multivar=False, channels=3, seed=42):
    np.random.seed(seed)
    if multivar:
        X = gen_multivar_data(n_samples, n_features, scale)
    else:
        X = np.random.normal(loc=0, scale=scale, size=(n_samples, n_features))

    noise = np.random.normal(loc=0, scale=noise_std, size=(n_samples, 1))
    X_tensor = torch.tensor(X).float()
    X_tensor = X_tensor.unsqueeze(-2).repeat(1, channels, 1)
    X = X_tensor.numpy()
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


def get_data(dataset_name):
    train_idx = None
    if dataset_name == 'gen_class':
        X, y, _, _ = generate_data(n_samples=1000, n_features=6, scale=3, noise_std=0.1, class_task=True)
        task = "class"
    elif dataset_name == 'gen_reg':
        X, y, _, _ = generate_data(n_samples=1000, n_features=6, scale=3, noise_std=0.1, class_task=False)
        task = "regr"
    elif dataset_name == 'gen_seq_class':
        X, y = generate_data_seq(n_samples=1000, n_features=16, scale=3, noise_std=0.1, class_task=True, channels=3)
        task = "class"
    elif dataset_name == 'gen_seq_reg':
        X, y = generate_data_seq(n_samples=1000, n_features=16, scale=3, noise_std=0.1, class_task=False, channels=3)
        task = "regr"
    elif dataset_name == 'bank':
        X, y = load_bank_marketing()
        task = "class"
    elif dataset_name == 'income':
        X, y = load_adult_income()
        task = "class"
    elif dataset_name == 'telescope':
        X, y = load_gamma_telescope()
        task = "class"
    elif dataset_name == 'maint':
        task = "class"
        X, y = load_maintenance()
    elif dataset_name == 'occupancy':
        task = "class"
        X, y, train_idx = load_occupancy()
    elif dataset_name == 'air':
        task = "regr"
        X, y = load_air_quality()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return X, y, train_idx, task
