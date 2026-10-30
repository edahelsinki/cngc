import time
import torch

from optimizer.pc_optimizer import GradientMeanEstimator, Preconditioner
from optimizer.opt_utils import upd_model_param
from optimizer.sophia import SophiaG, SophiaC
from training.grad_utils import get_grad_auto, get_noise, get_param_sizes, get_grad_ggn
from training.grad_utils import upd_sophia_hessian_ema, upd_cngc_hessian_ema
from training.train_utils import calc_dataloader_loss, calc_accuracy, calc_roc_auc, loss_print_formatter
from training.train_utils import set_lr, calc_acc_roc_auc_loader


class ModelTrainer:
    def __init__(
        self,
        n_epochs=10,
        lr=0.01,
        task='regr',
        use_pc=False,
        clean_updates=False,
        opt="gd",
        upd_every=None,
        use_blocks=False,
        noise_mult=1.0,
        binary_noise=False,
        subtract_clean=False,
        upd_every_ema=10,
        use_ggn=False,
    ):
        self.n_epochs = n_epochs
        self.lr = lr
        self.task = task
        self.use_pc = use_pc
        self.clean_updates = clean_updates
        self.opt = opt
        self.upd_every = upd_every
        self.use_blocks = use_blocks
        self.noise_mult = noise_mult
        self.binary_noise = binary_noise
        self.subtract_clean = subtract_clean
        self.upd_every_ema = upd_every_ema
        self.use_ggn = use_ggn

    def _get_optimizer(self, model):
        opt = self.opt.lower()
        if opt == 'gd':
            optim_cls = torch.optim.SGD
        elif opt == 'adam':
            optim_cls = torch.optim.Adam
        elif opt == 'muon':
            optim_cls = torch.optim.Muon
        elif opt == 'sophia':
            optim_cls = SophiaG
        elif opt == 'sophiac':
            optim_cls = SophiaC
        else:
            raise ValueError(f"Unknown optimizer: {opt}")

        lr = self.lr[0] if isinstance(self.lr, tuple) else self.lr
        return optim_cls(model.parameters(), lr=lr)

    def _get_lr_scheduler(self, optimizer):
        if isinstance(self.lr, tuple):
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[self.n_epochs // 2 + 1],
                gamma=self.lr[1] / self.lr[0]
            )
        return None

    def train(self, model, trainloader, testloader):
        class_task = self.task == 'class'
        print("Update every # batches", self.upd_every)

        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []
        wall_time_list = []

        optimizer = self._get_optimizer(model)
        lr_scheduler = self._get_lr_scheduler(optimizer)
        upd_hess_ema = isinstance(optimizer, SophiaG)

        grad_mean_calc = GradientMeanEstimator()
        param_sizes = get_param_sizes(model) if self.use_blocks else None
        cov_pc = Preconditioner(param_sizes=param_sizes, inverse_method="pinv",
                                noise_mult=self.noise_mult, use_ggn=self.use_ggn) if self.use_pc else None

        for epoch in range(self.n_epochs):
            model.train()
            loss_epoch = 0.0
            epoch_time = time.time()

            for batch_idx, batch in enumerate(trainloader):
                data, target, noise = batch
                model.zero_grad()

                if self.use_ggn:
                    # compute full jacobian to get GGN for the preconditioning
                    clean_grad, loss_clean, ggn = get_grad_ggn(model, data, target, class_task=class_task)
                    loss_epoch += loss_clean.item()
                    grad = clean_grad
                    if self.use_pc:
                        cov_pc.update(ggn)
                else:
                    params = list(model.parameters())
                    pred = model(data)

                    clean_grad, loss_clean = get_grad_auto(params, pred, target, class_task=class_task)
                    loss_epoch += loss_clean.item()

                    if self.use_pc or not self.clean_updates:
                        grad = self.get_noisy_grad(model, batch, clean_grad, pred, cov_pc, class_task)
                    else:
                        grad = clean_grad

                self.record_upd_grad(grad_mean_calc, clean_grad, grad)

                if (self.upd_every is not None and batch_idx % self.upd_every == 0) or batch_idx == len(trainloader) - 1:
                    grad_mean = grad_mean_calc.finalise()

                    if batch_idx == len(trainloader) // 2 and self.use_pc:
                        # update loss for the adapter more frequently
                        cov_pc.tau_adapter.record_f_prev(loss_epoch / (batch_idx + 1))

                    if batch_idx == len(trainloader) - 1 and self.use_pc:
                        cov_pc.finalise(grad=grad_mean)
                        cov_pc.tau_adapter.update(loss_epoch / len(trainloader))
                        set_lr(optimizer, cov_pc.alpha_star)

                    upd_model_param(model, optimizer, grad_mean, cov_pc, bs=data.shape[0])

                if upd_hess_ema and batch_idx % self.upd_every_ema == self.upd_every_ema - 1:
                    self.upd_sophia_ema(model, batch, optimizer, clean_grad, class_task)

            wall_time_i = time.time() - epoch_time
            wall_time_list.append(wall_time_i)

            if lr_scheduler is not None:
                lr_scheduler.step()
                print("Current lr is ", optimizer.param_groups[0]['lr'])

            loss_epoch = loss_epoch / len(trainloader)
            train_losses.append(loss_epoch)
            print(f'Epoch [{epoch + 1}/{self.n_epochs}], Loss: {loss_print_formatter(loss_epoch)}')
            if class_task:
                train_accuracies.append(calc_acc_roc_auc_loader(model, trainloader, is_test=False))

            if (epoch + 1) % 1 == 0:
                test_loss_epoch = calc_dataloader_loss(testloader, model, class_task=class_task)
                test_losses.append(test_loss_epoch)
                print(f'Test Loss: {loss_print_formatter(test_loss_epoch)}')
                if class_task:
                    test_accuracies.append(calc_acc_roc_auc_loader(model, testloader, is_test=True))

        metrics = {"train_losses": train_losses, "test_losses": test_losses,
                   "train_accuracies": train_accuracies, "test_accuracies": test_accuracies,
                   "wall_time": wall_time_list
                  }
        return model, metrics

    def upd_sophia_ema(self, model, batch, optimizer, clean_grad, class_task):
        data, target, noise = batch
        if isinstance(optimizer, SophiaC):
            model.zero_grad()
            pred = model(data)
            noisy_target = self.get_noisy_target(model, data, pred, target, noise, class_task)
            clen_grad_ema = clean_grad if self.subtract_clean else None
            upd_cngc_hessian_ema(model, optimizer, pred, noisy_target, clen_grad_ema, class_task=class_task)
        else:
            upd_sophia_hessian_ema(model, optimizer, data, class_task=class_task)

    def record_upd_grad(self, grad_mean_calc, clean_grad, grad):
        if self.clean_updates:
            grad_mean_calc.update(clean_grad)
        else:
            grad_mean_calc.update(grad)

    def get_noisy_grad(self, model, batch, clean_grad, pred, cov_pc, class_task: bool):
        data, target, noise = batch
        params = list(model.parameters())
        noisy_target = self.get_noisy_target(model, data, pred, target, noise, class_task)
        grad, _ = get_grad_auto(params, pred, noisy_target, class_task=class_task)

        if self.use_pc:
            grad_cov = grad - clean_grad if self.subtract_clean else grad
            cov_pc.update(grad_cov)
        return grad

    def get_noisy_target(self, model, data, pred, target, noise, class_task: bool):
        model.zero_grad()
        noise = get_noise(pred, target, data.shape[0], class_task=class_task,
                          noise=noise,
                          noise_mult=self.noise_mult,
                          binary_noise=self.binary_noise)
        noisy_target = target + noise
        noisy_target = noisy_target.detach()
        return noisy_target
