import argparse
import os
import time

# Set CUBLAS workspace config BEFORE importing torch
# This is critical for CUDA determinism
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from torch.utils.data import TensorDataset

from dataset.data_utils import get_dataload_with_seed
from dataset.data_utils import get_data
from models.model_utils import get_model
from utils.run_utils import get_seeds, fix_seed
from utils.estimators import (get_ngc, get_ggn_batch, get_hessian_batch, get_cgc, hutchinson_hessian_diagonal, get_gnb,
                              get_ngc_single_batch)
from utils.plot_utils import plot_ngc_comparison


def process_data(X, y, n_samples=200, class_task=False, norm_X=False, norm_y=False):
    X = torch.tensor(X[:n_samples]).float()
    y = torch.tensor(y[:n_samples]).float()

    # select axes that are reduced for normalization (preserve feature axis=1)
    if X.ndim == 2:
        axes = 0
    else:
        axes = tuple(i for i in range(X.ndim) if i != 1)

    X_mean = X.mean(dim=axes, keepdim=True)
    if norm_X:
        X_std = X.std(dim=axes, keepdim=True)
        X = (X - X_mean) / (X_std + 1e-8)
    else:
        X = X - X_mean

    if not class_task:
        y_mean = y.mean(dim=0)
        if norm_y:
            y_std = y.std(dim=0)
            y = (y - y_mean) / y_std
        else:
            y = y - y_mean

    return X, y


def compare_matr(gt_matr, matr):
    diff = torch.linalg.norm(gt_matr - matr, ord=2) / (torch.linalg.norm(gt_matr, ord=2) + 1e-8)

    # apply relu since the eigenvalues of the true Hessian can be negative
    eigv_gt_matr = torch.relu(torch.linalg.eigvalsh(gt_matr))
    eigv_matr = torch.relu(torch.linalg.eigvalsh(matr))
    diff_eigv = torch.linalg.norm(eigv_gt_matr - eigv_matr, ord=2) / (torch.linalg.norm(eigv_gt_matr, ord=2) + 1e-8)

    return diff, diff_eigv


def save_results(folder, filename, time, gt_matr, matr, verbose=False):
    if "hutch" in filename:
        # hutch estimator is only a diagonal
        matr = torch.eye(len(gt_matr)) * matr

    diff, diff_eigv = compare_matr(gt_matr, matr)
    metrics = {
        "time": time,
        "diff": diff,
        "diff_eigv": diff_eigv,
        "matr": matr,
    }
    res_path = os.path.join(folder, filename)
    np.savez(res_path, **metrics)

    if verbose:
        # print(f"Results saved to {res_path}")
        print(f"Time: {time:.4f} seconds")
        print(f"Rel. diff. ({filename}) matrices: {diff:.4f}")
        print(f"Rel. diff. ({filename}), eigenvalues: {diff_eigv:.4f}")
        print()

    return


def run_n_samples(model, X, y, n_samples, class_task, res_folder, device, args, verbose=False):
    subtract_clean = False if args.batch_size >= 64 else True
    noise_mult = 10
    inputs, targets = process_data(X, y, n_samples=n_samples, class_task=class_task, norm_X=args.norm_X)
    dataset = TensorDataset(inputs, targets)
    dataloader = get_dataload_with_seed(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ground truth GGN
    time_ggn1 = time.time()
    ggn = get_ggn_batch(model, dataloader, class_task=class_task, device=device)
    time_ggn2 = time.time()
    time_ggn = time_ggn2 - time_ggn1

    if args.use_h:
        print("Calculate Hessian")
        cpu = torch.device("cpu")
        model.to(cpu)
        gt_matr = get_hessian_batch(model, dataloader, class_task=class_task, device=cpu)
        model.to(device)
    else:
        gt_matr = ggn

    save_results(res_folder, f"ggn_{n_samples}.npz", time_ggn, gt_matr, ggn, verbose=verbose)

    # calibrated noisy grad outer product (no centering)
    time_ngc1 = time.time()
    ngc2 = get_ngc_single_batch(model, dataloader, class_task=class_task, noise_mult=noise_mult, subtract_clean=subtract_clean, device=device)
    time_ngc2 = time.time()
    time_ngc_unc = time_ngc2 - time_ngc1
    save_results(res_folder, f"ngc_unc_{n_samples}.npz", time_ngc_unc, gt_matr, ngc2, verbose=verbose)

    # uncalibrated gradient covariance
    time_ungc1 = time.time()
    ungc = get_ngc(model, dataloader, class_task=class_task, noise_mult=noise_mult, subtract_clean=False, calibrate=False, device=device)
    time_ungc2 = time.time()
    time_ungc = time_ungc2 - time_ungc1
    save_results(res_folder, f"ungc_{n_samples}.npz", time_ungc, gt_matr, ungc, verbose=verbose)

    # clean gradient covariance
    time_clean_ngc1 = time.time()
    cgc = get_cgc(model, dataloader, class_task=class_task, device=device)
    time_clean_ngc2 = time.time()
    time_clean_ngc = time_clean_ngc2 - time_clean_ngc1
    save_results(res_folder, f"cgc_{n_samples}.npz", time_clean_ngc, gt_matr, cgc, verbose=verbose)

    # Hutchinson estimator of the Hessian
    time_hutch1 = time.time()
    hutch = hutchinson_hessian_diagonal(model, dataloader, class_task=class_task, num_samples=1, device=device)
    time_hutch2 = time.time()
    time_hutch = time_hutch2 - time_hutch1
    save_results(res_folder, f"hutch_{n_samples}.npz", time_hutch, gt_matr, hutch, verbose=verbose)

    # Gauss-Newton-Bartlett estimator
    time_gnb1 = time.time()
    gnb = get_gnb(model, dataloader, class_task=class_task, use_mse=False, device=device)
    time_gnb2 = time.time()
    time_gnb = time_gnb2 - time_gnb1
    save_results(res_folder, f"gnb_{n_samples}.npz", time_gnb, gt_matr, gnb, verbose=verbose)

    # Gauss-Newton-Bartlett estimator
    time_gnb1 = time.time()
    gnb = get_gnb(model, dataloader, class_task=class_task, use_mse=True, device=device)
    time_gnb2 = time.time()
    time_gnb = time_gnb2 - time_gnb1
    save_results(res_folder, f"gnb_mse_{n_samples}.npz", time_gnb, gt_matr, gnb, verbose=verbose)

    # calibrated noisy grad covariance
    time_ngc1 = time.time()
    ngc = get_ngc(model, dataloader, class_task=class_task, noise_mult=noise_mult, subtract_clean=subtract_clean, device=device)
    time_ngc2 = time.time()
    time_ngc = time_ngc2 - time_ngc1
    save_results(res_folder, f"ngc_{n_samples}.npz", time_ngc, gt_matr, ngc, verbose=verbose)

    if verbose:
        print(f"Time GGN: {time_ggn}")
        print(f"Time NGC: {time_ngc}")
        print(f"Time NGC (uncentered): {time_ngc_unc}")
        print(f"Time UNGC: {time_ungc}")
        print(f"Time CGC: {time_clean_ngc}")
        print(f"Time Hutchinson: {time_hutch}")
        print(f"Time GNB: {time_gnb}")
    return


def inject_outliers(y, outlier_fraction=0.15, intensity=10.0, class_task=False):
    n_samples = len(y)
    n_outliers = int(n_samples * outlier_fraction)

    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    y_noisy = y.copy().astype(float)

    if class_task:
        y_noisy[outlier_indices] = 1 - y_noisy[outlier_indices]
    else:
        std_dev = np.std(y)
        direction = np.random.choice([-1, 1], size=n_outliers)
        shift = direction * (intensity * std_dev)
        y_noisy[outlier_indices] += shift

    return y_noisy


def run(res_folder_seed, n_batches, args):
    X, y, train_idx, task = get_data(args.dataset)
    class_task = task == 'class'
    if args.dataset == "air":
        # double the dataset to be able to test for larger number of samples, since the original dataset a bit below 10k
        X = np.concatenate([X, X], axis=0)
        y = np.concatenate([y, y], axis=0)

    if args.inject_outliers:
        print("Injecting outliers in the targets...")
        y = inject_outliers(y, outlier_fraction=0.05, intensity=5.0, class_task=class_task)

    y = y.reshape(-1, 1)
    model = get_model(args.model, X.shape, y.shape, args.dataset, hidden_size=args.hidden_size)
    print(f"Number of parameters: {model.get_n_params()}")
    print(f"Number of dataset samples is {len(X)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for n in n_batches:
        n_samples = n * args.batch_size
        print(f"n_samples: {n_samples}")
        run_n_samples(model, X, y, n_samples, class_task, res_folder_seed, device, args, verbose=args.verbose)
    return


def plot_ngc_res(res_folder, seeds, n_batches, batch_size, log_scale=False):
    n_batches = n_batches[1:]
    res_time = {'ggn': [], 'ungc': [], 'cgc': [], 'hutch': [], 'gnb': [], 'gnb_mse': [], 'ngc': []}
    res_diff = {'ggn': [], 'ungc': [], 'cgc': [], 'hutch': [], 'gnb': [], 'gnb_mse': [], 'ngc': []}
    res_diff_eigv = {'ggn': [], 'ungc': [], 'cgc': [], 'hutch': [], 'gnb': [], 'gnb_mse': [], 'ngc': []}

    for n in n_batches:
        n_samples = n * batch_size
        print(f"n_samples: {n_samples}")
        for est in res_time.keys():
            time, diff, diff_eigv = [], [], []

            for seed in seeds:
                res_folder_seed = os.path.join(res_folder, str(seed))
                filename = f"{est}_{n_samples}.npz"
                res_path = os.path.join(res_folder_seed, filename)
                if os.path.exists(res_path):
                    data = np.load(res_path)
                    time.append(data['time'])
                    diff.append(data['diff'])
                    diff_eigv.append(data['diff_eigv'])
                else:
                    print(f"Missing results for {filename} for seed {seed}.")
                    time.append(np.nan)
                    diff.append(np.nan)
                    diff_eigv.append(np.nan)

            res_time[est].append(time)
            res_diff[est].append(diff)
            res_diff_eigv[est].append(diff_eigv)

    # plot the results
    plot_ngc_comparison(res_time, n_batches, batch_size, res_folder, log_scale=False, use_h=args.use_h, title="time")
    plot_ngc_comparison(res_diff, n_batches, batch_size, res_folder, log_scale=log_scale, use_h=args.use_h, title="full_matr_diff")
    plot_ngc_comparison(res_diff_eigv, n_batches, batch_size, res_folder, log_scale=log_scale, use_h=args.use_h, title="eigv_diff")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=0, help='Number of random seeds to average over.')
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument('--dataset', type=str,
                        choices=['gen_reg', 'gen_class', 'gen_seq_reg', 'gen_seq_class', 'bank', "income",
                                 "telescope", "occupancy", "air"],
                        help='The name of the dataset.')
    parser.add_argument('--model', type=str, choices=['linear', 'mlp', 'cnn', 'transf'],
                        help='The type of the model.')
    parser.add_argument('--hidden_size', type=int, default=20, help='Size of MLP or CNN layers.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument("--norm_X", action='store_true',
                        help="Flag to divide X by std (mean is subtracted in any case).")
    parser.add_argument("--use_h", action='store_true',
                        help="Flag to indicate that the true Hessian is used as a ground truth.")
    parser.add_argument("--res_folder_root", type=str, default='results_ngc_comparison',
                        help="Root folder to save results")
    parser.add_argument("--verbose", action='store_true',
                        help="Flag to print the results after each estimator and number of samples.")
    parser.add_argument("--inject_outliers", action='store_true',
                        help="Flag to inject outliers in the training targets.")
    parser.add_argument("--plot_res", action='store_true',
                        help="Flag to plot the final results")
    parser.add_argument("--plot_only", action='store_true',
                        help="Flag to plot the results without training for missing runs")
    args = parser.parse_args()

    seed = args.seed
    fix_seed(seed)

    res_folder_root = args.res_folder_root if args.res_folder_root else 'results_ngc_comparison'
    model_task_folder = f"{args.model}_{args.dataset}_outl" if args.inject_outliers else f"{args.model}_{args.dataset}"
    if args.batch_size != 128:
        model_task_folder += f"_bs_{args.batch_size}"

    if args.use_h:
        model_task_folder += "_h_gt"

    res_folder = os.path.join(os.getcwd(), res_folder_root, model_task_folder)
    print(res_folder)
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    if args.n_seeds > 0:
        seeds = get_seeds(args.n_seeds)
    else:
        seeds = [args.seed]
    print(f"Using seeds: {seeds}")

    n_batches = [20, 40, 60, 80, 100, 120, 140]
    if not args.plot_only:
        for seed in seeds:
            print(f"\nRunning with seed {seed}")
            # crete a subfolder for the seed
            res_folder_seed = os.path.join(res_folder, f"{seed}")
            if not os.path.exists(res_folder_seed):
                os.makedirs(res_folder_seed)

            files = os.listdir(res_folder_seed)

            # Run the evaluation
            run(res_folder_seed, n_batches, args)

    if args.plot_res:
        plot_ngc_res(res_folder, seeds, n_batches, args.batch_size, log_scale=True)
