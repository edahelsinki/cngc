import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from training.grad_utils import calc_loss


def calc_dataloader_loss(testloader, model, class_task=False):
    model.eval()
    test_loss_total = 0.0
    for data, target, _ in testloader:
        with torch.no_grad():
            outputs = model(data)
            test_loss = calc_loss(outputs, target, class_task=class_task)
            test_loss_total += test_loss.item()

    test_loss_total /= len(testloader)
    return test_loss_total


def calc_accuracy(model, testloader):
    model.eval()
    correct = 0
    total = 0
    for data, target, _ in testloader:
        with torch.no_grad():
            output = model(data)
            output = torch.sigmoid(output)
            predicted = (output > 0.5).float()
            correct += (predicted == target).float().sum().item()
            total += target.size(0)
    accuracy = correct / total
    return accuracy


def calc_roc_auc(model, loader):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch, _ in loader:
            logits = model(X_batch).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_batch.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    return auc


def calc_acc_roc_auc_loader(model, dataloader, is_test=False):
    acc = calc_accuracy(model, dataloader)
    roc_auc = calc_roc_auc(model, dataloader)
    split_name = 'Test' if is_test else 'Train'
    print(f'{split_name} Accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}')
    return acc


def loss_print_formatter(loss):
    loss_fmt = f"{loss:.4f}" if np.abs(loss) > 0.0001 else f"{loss:.4e}"
    return loss_fmt


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_lr_mult(optimizer, lr_mult):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_mult
