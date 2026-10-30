import os
import numpy as np
import copy
import torch


# To ensure reproducibility on the cluster with np.linalg.solve (for the analytical solution)
# torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# switch off flash attention and memory efficient attention to avoid issues with backward pass with vmap and hessian calc
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def generate_data(n_samples=100000, n_features=10, scale=1, noise_std=0.1, w=None, b=None, multivar=False):
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
    return X, y, w, b


def generate_data_nn(model, n_samples=100000, n_features=10, scale=1, noise_std=0.1, class_task=False, multivar=False,
                     seed=42):
    np.random.seed(seed)
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
    print("Model distortion with norm:", dist_norm_w)
    total_params = sum(p.numel() for p in model_gen.parameters())
    print(f"Total number of model parameters: {total_params}")

    # distort the model weights
    model = copy.deepcopy(model_gen)
    model_dist = copy.deepcopy(model_gen)
    sensi_w = get_param_sensitivity(model, X, y) if X is not None and y is not None else None

    w_distortions = model.get_param_distortion(sensitivities=sensi_w)
    model_dist.set_params(w_distortions, distortion_norm_w=dist_norm_w, distortion_norm_b=0.1)

    if X is not None and y is not None:
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        else:
            X_tensor = X

        if isinstance(y, np.ndarray):
            y_tensor = torch.from_numpy(y[:, None]).float()
        else:
            y_tensor = y

        pred = model_dist(X_tensor)
        mse_dist = torch.mean((pred - y_tensor)**2).item()
        mse_gt = torch.mean((model_gen(X_tensor) - y_tensor)**2).item()
        print(f"MSE with GT model: {mse_gt:.4f}")
        print(f"MSE with distorted model: {mse_dist:.4f}")
    return model_dist


def eval_loss(model, X, y, loss_fn):
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y[:, None], dtype=torch.float32)
    with torch.no_grad():
        return loss_fn(model(X), y).item()


def get_param_sensitivity(model, X, y, eps=1e-3, loss_fn=torch.nn.MSELoss()):
    baseline_loss = eval_loss(model, X, y, loss_fn)

    sensitivities = {}
    for name, param in model.named_parameters():
        rand_vec = torch.randn_like(param)
        rand_vec = rand_vec / torch.linalg.norm(rand_vec)

        with torch.no_grad():
            param.add_(eps * rand_vec)
        perturbed_loss = eval_loss(model, X, y, loss_fn)
        with torch.no_grad():
            param.sub_(eps * rand_vec)

        sensitivities[name] = (perturbed_loss - baseline_loss) / eps
    return sensitivities
