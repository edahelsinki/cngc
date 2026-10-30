import os
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

from dataset.data_utils import get_dataload_with_seed
from utils.run_utils import get_seeds, fix_seed
from utils.gen_data_utils import generate_data_nn
from utils.gen_data_utils import distort_model
from models.mlp import MLPRegressor
from utils.estimators import get_ggn_batch, get_ngc_single_batch, get_ngc
from utils.plot_utils import plot_gb_noise_results
from utils.matr_comparison_utils import get_matr_diff


def run_dist_w(mlp_dict, X, y, batch_size=128, class_task=True,
               filepath="results_gb_noise/gb_noise_dist_w.npz"):
    X = torch.tensor(X).float()
    y = torch.tensor(y[:, None]).float()

    dataset = TensorDataset(X, y)
    dataloader = get_dataload_with_seed(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    dist_norm_w_list = list(mlp_dict.keys())
    n_points = len(dist_norm_w_list)
    binary_diff_array = np.zeros((1, n_points))
    gaussian_diff_array = np.zeros((1, n_points))

    for i, dist_norm_w in enumerate(dist_norm_w_list):
        model_dist = mlp_dict[dist_norm_w]
        print(f"Evaluating: dist_norm_w {dist_norm_w}")
        # get GGN
        ggn = get_ggn_batch(model_dist, dataloader, class_task=class_task)

        # GGN est centered by subtracting the clean gradient, calculation on the same batch
        single_batch = get_ngc_single_batch(model_dist, dataloader, class_task=class_task, noise_mult=1.,
                                            subtract_clean=True, binary_noise=True)
        single_diff = get_matr_diff(ggn, single_batch)
        binary_diff_array[0, i] = single_diff

        single_batch = get_ngc_single_batch(model_dist, dataloader, class_task=class_task, noise_mult=1.,
                                            subtract_clean=True, binary_noise=False)
        single_diff = get_matr_diff(ggn, single_batch)
        gaussian_diff_array[0, i] = single_diff
        print()

    np.savez(filepath, binary_diff=binary_diff_array, gaussian_diff=gaussian_diff_array, dist_list=dist_norm_w_list)
    return binary_diff_array, gaussian_diff_array


def run_est_noise(mlp_dict, X_all, y_all):
    filepath = f"{res_folder_seed}/gb_noise_dist_w.npz"
    if os.path.exists(filepath):
        data = np.load(filepath)
        binary_diff = data["binary_diff"]
        gaussian_diff = data["gaussian_diff"]
        dist_list = data['dist_list']
    else:
        fix_seed(seed)
        dist_list = list(mlp_dict.keys())
        binary_diff, gaussian_diff = run_dist_w(mlp_dict, X_all[:n_samples], y_all[:n_samples],
                                             batch_size=batch_size, class_task=class_task, filepath=filepath)

    if args.plot_res:
        fig, ax1 = plt.subplots(1, 1, figsize=(3.15, 2), sharey=True)
        plot_gb_noise_results(ax1, dist_list, binary_diff, gaussian_diff, "Norm of param distortion", "", set_ylabel=True)

        plt.tight_layout()
        plt.savefig(f"{res_folder_seed}/gb_noise_test_{seed}.pdf")
        plt.show()


def plot_gb_noise_res(res_folder, seeds):
    binary_diff_list, gaussian_diff_list = [], []

    dist_axis = None
    for seed in seeds:
        res_folder_seed = os.path.join(res_folder, f"{seed}")

        filepath = os.path.join(res_folder_seed, f"gb_noise_dist_w.npz")
        if os.path.exists(filepath):
            data = np.load(filepath)
            binary_diff_list.append(data["binary_diff"])
            gaussian_diff_list.append(data["gaussian_diff"])
            dist_axis = data['dist_list']
        else:
            print(f"!!! Missing results for dist_norm_w for seed {seed}")

    binary_diff = np.concatenate(binary_diff_list, axis=0)
    gaussian_diff = np.concatenate(gaussian_diff_list, axis=0)

    fig, ax1 = plt.subplots(1, 1, figsize=(3.15, 2), sharey=True)
    plot_gb_noise_results(ax1, dist_axis, binary_diff, gaussian_diff, "Norm of param distortion", "", set_ylabel=True)

    plt.tight_layout()
    plt.savefig(os.path.join(res_folder, "gb_noise_test.pdf"), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=0, help='Number of random seeds to average over.')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--res_folder_root", type=str, default='results_gb_noise',
                        help="Root folder to save results")
    parser.add_argument("--plot_res", action='store_true',
                        help="Flag to plot the final results")
    parser.add_argument("--plot_only", action='store_true',
                        help="Flag to plot the results without training for missing runs")

    args = parser.parse_args()
    res_folder_root = args.res_folder_root if args.res_folder_root else 'results_gb_noise'
    res_folder = os.path.join(os.getcwd(), res_folder_root)
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    if args.n_seeds > 0:
        seeds = get_seeds(args.n_seeds)
    else:
        seeds = [args.seed]
    print(f"Using seeds: {seeds}")

    if not args.plot_only:
        n_features = 20
        batch_size = 64
        class_task = False
        n_samples = 50000

        # generate data once and then use it for every seed for noise sampling
        fix_seed(0)
        # create a model, generate data, and distort the model once, to be used for all experiments
        mlp = MLPRegressor(in_features=n_features, out_features=1, hidden_layers=[5], add_bias=False, is_relu=True)
        X_all, y_all = generate_data_nn(mlp, n_samples=n_samples, n_features=n_features, noise_std=0.01,
                                        scale=1, multivar=True, class_task=class_task, seed=0)

        dist_list = np.arange(0.1, 3.0, 0.5)
        model_dict = {}
        for dist_norm_w in dist_list:
            model_dist = distort_model(mlp, dist_norm_w, X=X_all, y=y_all)
            model_dict[dist_norm_w] = model_dist

        for seed in seeds:
            print(f"\nRunning with seed {seed}")
            fix_seed(seed)

            res_folder_seed = os.path.join(res_folder, f"{seed}")
            if not os.path.exists(res_folder_seed):
                os.makedirs(res_folder_seed)

            run_est_noise(model_dict, X_all, y_all)

    if args.plot_res:
        plot_gb_noise_res(res_folder, seeds)
