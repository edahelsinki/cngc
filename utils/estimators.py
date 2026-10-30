import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap, jacfwd

from training.grad_utils import get_grad_auto, get_noise
from optimizer.pc_optimizer import GradientCovarianceEstimator


def get_ngc(model, dataloader, class_task=False, noise_mult=10., subtract_clean=True, calibrate=True, device=None):
    if device is None:
        device = torch.device("cpu")

    cov_est = GradientCovarianceEstimator()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        model.zero_grad()
        params = list(model.parameters())
        pred = model(data)

        noise = get_noise(pred, target, data.shape[0], class_task=class_task,
                          noise_mult=noise_mult,
                          binary_noise=True,
                          calibrate=calibrate)

        noisy_target = target + noise
        noisy_target = noisy_target.detach()
        grad, _ = get_grad_auto(params, pred, noisy_target, class_task=class_task)

        if subtract_clean:
            model.zero_grad()
            clean_grad, _ = get_grad_auto(params, pred, target, class_task=class_task)
            grad_cov = grad - clean_grad
        else:
            grad_cov = grad

        cov_est.update(grad_cov)
    ngc = cov_est.finalise()

    if noise_mult > 0.:
        ngc = (1. / noise_mult**2) * ngc
    ngc = ngc.detach().cpu()
    return ngc


def get_cgc(model, dataloader, class_task=False, device=None):
    if device is None:
        device = torch.device("cpu")

    cov_est = GradientCovarianceEstimator()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        params = list(model.parameters())
        pred = model(data)
        clean_grad, _ = get_grad_auto(params, pred, target, class_task=class_task)
        cov_est.update(clean_grad)

    cgc = cov_est.finalise()
    cgc = cgc.detach().cpu()
    return cgc


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


def get_ggn_batch(model, dataloader, class_task=False, device=None):
    if device is None:
        device = torch.device("cpu")

    ggn = None
    for batch_idx, (inputs, target) in enumerate(dataloader):
        inputs = inputs.to(device)
        ggn_batch = get_full_ggn(model, inputs, class_task=class_task)

        if batch_idx == 0:
            ggn = ggn_batch
        else:
            ggn += ggn_batch

    ggn = ggn / len(dataloader)
    ggn= ggn.detach().cpu()
    return ggn


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
            # extract linear layers with the provided idx
            # the idx is multiplied by 2 because ReLU layers are also counted in the model
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


def get_hessian_batch(model, dataloader, class_task=False, device=None):
    if device is None:
        device = torch.device("cpu")

    hessian = None
    for batch_idx, (inputs, target) in enumerate(dataloader):
        inputs, target = inputs.to(device), target.to(device)
        hessian_batch = get_hessian(model, inputs, target, class_task=class_task)

        if batch_idx == 0:
            hessian = hessian_batch
        else:
            hessian += hessian_batch
    hessian = hessian / len(dataloader)
    hessian = hessian.detach().cpu()
    return hessian


def hutchinson_hessian_diagonal(model, dataloader, class_task=False, num_samples=100, device=None):
    if device is None:
        device = torch.device("cpu")

    model.eval()
    criterion = nn.BCEWithLogitsLoss() if class_task else nn.MSELoss()
    params = [p for p in model.parameters() if p.requires_grad]
    diag_accumulator = [torch.zeros_like(p) for p in params]

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_diag = [torch.zeros_like(p) for p in params]

        for _ in range(num_samples):
            # Forward pass — keep graph for double backprop
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # compute gradients (retain graph for Hv step)
            grads = torch.autograd.grad(loss, params, create_graph=True)

            # sample Rademacher vector (±1)
            v = [torch.randint_like(p, high=2).float() * 2 - 1 for p in params]

            # Hessian-vector product Hv
            Hv = torch.autograd.grad(
                outputs=grads,
                inputs=params,
                grad_outputs=v,
                retain_graph=False,
            )

            # Accumulate v ⊙ Hv  (unbiased estimator of diag(H))
            for i, (vi, hvi) in enumerate(zip(v, Hv)):
                batch_diag[i] += vi * hvi.detach()

        # Average over Rademacher samples and add to dataset accumulator
        for i in range(len(params)):
            diag_accumulator[i] += batch_diag[i] / num_samples

    # flatten the accumulated diagonals into a single vector
    diag_accumulator = [d.flatten() for d in diag_accumulator]
    diag = torch.cat(diag_accumulator) / len(dataloader)
    diag = diag.detach().cpu()
    return diag


def get_gnb(model, dataloader, class_task=False, use_mse=False, device=None):
    if device is None:
        device = torch.device("cpu")

    gnb = None
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        params = list(model.parameters())
        pred = model(data)

        if class_task:
            pred_probs = torch.sigmoid(pred).detach()
            samp_dist = torch.distributions.Bernoulli(probs=pred_probs)
            pred_sample = samp_dist.sample()
        else:
            if use_mse:
                loss = torch.nn.functional.mse_loss(pred, target, reduction="mean")
                std = torch.sqrt(loss).detach()
                # print(f"std is {std.item():.4f}")
            else:
                std = 1.
            pred_sample = pred.detach() + std * torch.randn_like(pred)

        grad, _ = get_grad_auto(params, pred, pred_sample, class_task=class_task)
        gnb_i = torch.outer(grad, grad) * data.shape[0]

        if batch_idx == 0:
            gnb = gnb_i
        else:
            gnb += gnb_i
    gnb = gnb / len(dataloader)
    gnb = gnb.detach().cpu()
    return gnb


def get_ngc_single_batch(model, dataloader, class_task=False, noise_mult=10., subtract_clean=True, calibrate=True,
                         binary_noise=True, device=None):
    if device is None:
        device = torch.device("cpu")

    ngc = None
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        params = list(model.parameters())
        pred = model(data)

        noise = get_noise(pred, target, data.shape[0], class_task=class_task,
                          noise_mult=noise_mult,
                          binary_noise=binary_noise,
                          calibrate=calibrate)

        noisy_target = target + noise
        noisy_target = noisy_target.detach()
        grad, _ = get_grad_auto(params, pred, noisy_target, class_task=class_task)

        if subtract_clean:
            model.zero_grad()
            clean_grad, _ = get_grad_auto(params, pred, target, class_task=class_task)
            grad_cov = grad - clean_grad
        else:
            grad_cov = grad

        if batch_idx == 0:
            ngc = torch.outer(grad_cov, grad_cov)
        else:
            ngc += torch.outer(grad_cov, grad_cov)
    ngc = ngc / len(dataloader)

    if noise_mult > 0.:
        ngc = (1. / noise_mult**2) * ngc
    ngc = ngc.detach().cpu()
    return ngc
