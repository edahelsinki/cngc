from optimizer.sophia import SophiaG

def upd_model_param(model, optimizer, grad_mean, cov_pc=None, bs=None):
    if cov_pc is not None:
        grad_mean = cov_pc.apply(grad_mean)

    optimizer.zero_grad()
    current_idx = 0
    for param in model.parameters():
        num_elements = param.numel()
        grad_slice = grad_mean[current_idx: current_idx + num_elements]
        current_idx += num_elements

        param.grad = grad_slice.view(param.shape)

    if isinstance(optimizer, SophiaG):
        optimizer.step(bs=bs)
    else:
        optimizer.step()
