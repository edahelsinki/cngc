from optimizer.pc_optimizer import GradientMeanEstimator, Preconditioner
from optimizer.opt_utils import upd_model_param
from optimizer.sophia import SophiaG
from training.grad_utils import get_grad_auto, get_param_sizes
from training.train_utils import calc_dataloader_loss, calc_acc_roc_auc_loader, loss_print_formatter
from training.train_utils import set_lr
from training.trainer import ModelTrainer
from optimizer.early_stop_stats import CredibleValue


class ModelTrainerWithEs(ModelTrainer):
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
        use_clean_es=False
    ):
        super().__init__(
            n_epochs=n_epochs,
            lr=lr,
            task=task,
            use_pc=use_pc,
            clean_updates=clean_updates,
            opt=opt,
            upd_every=upd_every,
            use_blocks=use_blocks,
            noise_mult=noise_mult,
            binary_noise=binary_noise,
            subtract_clean=subtract_clean,
            upd_every_ema=upd_every_ema
        )
        self.use_clean_es = use_clean_es

    def train(self, model, trainloader, testloader, valloader=None):
        class_task = self.task == 'class'
        print("Update every # batches", self.upd_every)

        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []
        es_stats = []

        optimizer = self._get_optimizer(model)
        lr_scheduler = self._get_lr_scheduler(optimizer)
        upd_hess_ema = isinstance(optimizer, SophiaG)

        grad_mean_calc = GradientMeanEstimator()
        param_sizes = get_param_sizes(model) if self.use_blocks else None
        cov_pc = Preconditioner(param_sizes=param_sizes, inverse_method="pinv", noise_mult=self.noise_mult) if self.use_pc else None

        # use either validation set or credible value as stopping criterion
        if valloader is None:
            cr_value = CredibleValue(param_sizes, n_batches=len(trainloader), use_clean_grad=self.use_clean_es,
                                     use_pc=self.use_pc, noise_mult=self.noise_mult)
        else:
            cr_value = None
        for epoch in range(self.n_epochs):
            model.train()
            loss_epoch = 0.0

            for batch_idx, batch in enumerate(trainloader):
                data, target, noise = batch
                model.zero_grad()
                params = list(model.parameters())
                pred = model(data)

                clean_grad, loss_clean = get_grad_auto(params, pred, target, class_task=class_task)
                loss_epoch += loss_clean.item()

                grad = self.get_noisy_grad(model, batch, clean_grad, pred, cov_pc, class_task)
                self.record_upd_grad(grad_mean_calc, clean_grad, grad)

                if cr_value is not None:
                    # update stats for early stopping with a covariance matrix
                    cr_value.record_grad(clean_grad, grad)

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

                    # calculate statistics for the early stopping criteria
                    if batch_idx == len(trainloader) - 1 and cr_value is not None:
                        es_stat = cr_value.get_value(cov_pc)
                        es_stats.append(es_stat)
                        print(f"ES stat: {es_stat:.4f}")

                if upd_hess_ema and batch_idx % self.upd_every_ema == self.upd_every_ema - 1:
                    self.upd_sophia_ema(model, batch, optimizer, clean_grad, class_task)

            if lr_scheduler is not None:
                lr_scheduler.step()
                print("Current lr is ", optimizer.param_groups[0]['lr'])

            # register validation loss if valloader is provided
            if valloader is not None:
                val_loss_epoch = calc_dataloader_loss(valloader, model, class_task=class_task)
                print(f'Validation Loss: {loss_print_formatter(val_loss_epoch)}')
                es_stats.append(val_loss_epoch)

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
                   "es_stats": es_stats}
        return model, metrics
