import numpy as np
import torch
import torch.functional as F
import torch.autograd as autograd

from torch.func import functional_call, jacrev, vmap, jacfwd


def get_param_sizes(model) -> list[int]:
    param_sizes = [p.numel() for p in model.parameters()]
    return param_sizes


def get_grad_auto(params, pred, target, class_task=False):
    loss = calc_loss(pred, target, class_task=class_task)
    grad = autograd.grad(loss, params, retain_graph=True)
    grad = concat_grad(grad)
    return grad, loss


def get_grad_ggn(model, inputs, target, class_task=False):
    # [B, P] for regression and [B, C, P] for classification, C = 1 (both for regression and for classification)
    jac, y_out = get_full_jac_no_sort(model, inputs, class_task=class_task)
    # y_out = model(inputs)  # [B, C] or [B]

    loss = calc_loss(y_out, target, class_task=class_task)
    batch_size = inputs.shape[0]

    if class_task:
        pred = torch.sigmoid(y_out)  # [B, C]

        # dL/dy_out = (pred - target) / B  -> use with jac [B, C, P]
        flat_jac = jac.reshape(batch_size, -1)  # [B, P]

        # Gradient: J^T @ delta / B
        delta = (pred - target)  # [B, C]
        grad = (flat_jac.T @ delta).squeeze(-1) / batch_size  # [P]

        # for classification the GGN is J^T V J where V is the diagonal matrix with v_i = p_i * (1 - p_i)
        v = (pred * (1. - pred)).detach()

        # # Scale jac by v directly (no sqrt!): [B, P] * [B, 1] -> [B, P]
        v_jac = flat_jac * v

        # (v * J)^T @ J  --> [P, B] @ [B, P] -> [P, P]
        ggn = (v_jac.T @ flat_jac)  # [P, P]

        # this is equivalent to the above but slower and uses more memory
        # v_mat = torch.diag_embed(v)  # [B, C, C]
        # ggn = jac.transpose(1, 2) @ v_mat @ jac  # [B, P, P]
        # ggn = torch.sum(ggn, dim=0)
    else:
        # dL/dy_out = (y_out - target) / B  -> use with jac [B, P]
        delta = (y_out - target)  # [B]
        grad = jac.T @ delta / batch_size
        grad = grad.reshape(-1)

        # for regression the GGN is J^T J
        # jac.T - [P, B] and jac is [B, P]
        ggn = jac.T @ jac  # [P, P]

    ggn  = ggn / batch_size
    return grad, loss, ggn


def concat_grad(grad):
    grad = torch.cat([g_i.flatten() for g_i in grad], dim=0)
    return grad


def calc_loss(y_out, y, class_task=False):
    if class_task:
        # for classification task, we can use cross-entropy loss
        pred = torch.sigmoid(y_out)
        ce = -torch.mean(y * torch.log(pred + 1e-8) + (1. - y) * torch.log(1. - pred + 1e-8))
        return ce
    else:
        mse = 0.5 * torch.mean((y_out - y)**2)
        return mse

def get_noise(y_pred, y_batch, batch_size, class_task=False, noise=None, noise_mult=1.0, binary_noise=False, calibrate=True):
    # if noise if saved in the dataloader, use it, otherwise generate new noise
    if noise is None or torch.sum(torch.abs(noise)) < 1e-5:
        if binary_noise:
            noise_unit = torch.randint(0, 2, size=y_batch.shape, device=y_batch.device).float() * 2 - 1
        else:
            noise_unit = torch.randn(size=y_batch.shape, device=y_batch.device)
        noise = noise_unit * np.sqrt(batch_size)

    if class_task and calibrate:
        v = torch.sqrt(torch.sigmoid(y_pred) * (1. - torch.sigmoid(y_pred))).detach()
        noise = noise * v

    noise = noise * noise_mult

    return noise


def upd_sophia_hessian_ema(model, optimizer, data, class_task=True):
    # update hessian EMA
    pred = model(data)

    if class_task:
        pred_probs = torch.sigmoid(pred).detach()
        samp_dist = torch.distributions.Bernoulli(probs=pred_probs)
        pred_sample = samp_dist.sample()
    else:
        std = 1.
        pred_sample = pred.detach() + std * torch.randn_like(pred)

    loss = calc_loss(pred, pred_sample, class_task=class_task)
    loss.backward()
    optimizer.update_hessian()
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad()


def upd_cngc_hessian_ema(model, optimizer, pred, noisy_target, clean_grad=None, class_task=True):
    loss = calc_loss(pred, noisy_target, class_task=class_task)
    loss.backward()

    if clean_grad is not None:
        for param in model.parameters():
            if param.grad is not None:
                param.grad -= clean_grad[:param.numel()].view(param.shape)
                clean_grad = clean_grad[param.numel():]
    optimizer.update_hessian()
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad()


def get_full_jac_no_sort(model, inputs, class_task=False, randomness="different"):
    """
    Computes the gradient of the model's output with respect to its parameters,
    and returns the output of that SAME forward pass (so dropout masks etc. match).
    """
    def fmodel(params, inputs):
        out = functional_call(model, params, inputs)
        return out, out  # (primal for jacrev, aux value we get back untouched)

    # Jacobian wrt params for ONE sample, plus the output from that same pass
    def single_jac(params, x):
        return jacrev(fmodel, has_aux=True)(params, x)  # -> (jac, out)

    def flatten_jacobian_mse(jac):
        B = next(iter(jac.values())).shape[0]
        mats = []
        for k in jac.keys():
            mats.append(jac[k].reshape(B, -1))
        return torch.cat(mats, dim=-1)  # [B, P]

    def flatten_jacobian_ce(jac):
        B, C = next(iter(jac.values())).shape[:2]
        mats = []
        for k in jac.keys():
            mats.append(jac[k].reshape(B, C, -1))  # [B, C, P]
        return torch.cat(mats, dim=-1)

    def flatten_jacobian(jac, class_task):
        if class_task:
            return flatten_jacobian_ce(jac)
        else:
            return flatten_jacobian_mse(jac)

    # Vectorize over batch
    params = dict(model.named_parameters())
    jac, y_out = vmap(single_jac, in_dims=(None, 0), randomness=randomness)(params, inputs)

    # flatten Jacobian
    jac_matrix = flatten_jacobian(jac, class_task) # B, P for MSE, B, C, P for CE (if C > 1)
    return jac_matrix, y_out
