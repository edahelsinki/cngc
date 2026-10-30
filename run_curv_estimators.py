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
from utils.plot_utils import plot_curv_est_results
from utils.matr_comparison_utils import get_matr_diff


def run_dist_w(mlp, X, y,dist_norm_w_list, batch_size=128, class_task=True,
               filepath="results_curv/noise_dist_w.npz"):
    X = torch.tensor(X).float()
    y = torch.tensor(y[:, None]).float()

    dataset = TensorDataset(X, y)
    dataloader = get_dataload_with_seed(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    n_points = len(dist_norm_w_list)
    single_diff_array = np.zeros((1, n_points))
    cross_diff_array = np.zeros((1, n_points))

    for i, dist_norm_w in enumerate(dist_norm_w_list):
        model_dist = distort_model(mlp, dist_norm_w, X=X, y=y)
        print(f"Evaluating: dist_norm_w {dist_norm_w}")
        # get GGN
        ggn = get_ggn_batch(model_dist, dataloader, class_task=class_task)

        # GGN est is "uncentered" since the clean grad is  not subtracted, so mean is calculated across batches
        cross_batch = get_ngc(model_dist, dataloader, class_task=class_task, noise_mult=1., subtract_clean=False)
        cross_diff = get_matr_diff(ggn, cross_batch)
        cross_diff_array[0, i] = cross_diff

        # GGN est centered by subtracting the clean gradient, calculation on the same batch
        single_batch = get_ngc_single_batch(model_dist, dataloader, class_task=class_task, noise_mult=1., subtract_clean=True)
        single_diff = get_matr_diff(ggn, single_batch)
        single_diff_array[0, i] = single_diff
        print()

    np.savez(filepath, single_diff=single_diff_array, cross_diff=cross_diff_array, dist_list=dist_norm_w_list)
    return single_diff_array, cross_diff_array


def run_curv_est():
    # create a model, generate data, and distort the model once, to be used for all experiments
    mlp = MLPRegressor(in_features=n_features, out_features=1, hidden_layers=[5], add_bias=False, is_relu=True)
    X_all, y_all = generate_data_nn(mlp, n_samples=n_samples, n_features=n_features, noise_std=0.01,
                                    scale=1, multivar=True, class_task=class_task, seed=seed)

    filepath = f"{res_folder_seed}/curv_est_dist_w.npz"
    if os.path.exists(filepath):
        data = np.load(filepath)
        single_diff = data["single_diff"]
        cross_diff = data["cross_diff"]
        dist_list = data['dist_list']
    else:
        fix_seed(seed)
        dist_list = np.arange(0.1, 3.0, 0.5)
        single_diff, cross_diff = run_dist_w(mlp, X_all[:n_samples], y_all[:n_samples], dist_list,
                                             batch_size=batch_size, class_task=class_task, filepath=filepath)

    if args.plot_res:
        fig, ax1 = plt.subplots(1, 1, figsize=(3, 2), sharey=True)
        plot_curv_est_results(ax1, dist_list, single_diff, cross_diff, "Norm of param distortion", "", set_ylabel=True)

        plt.tight_layout()
        plt.savefig(f"{res_folder_seed}/curv_est_test_{seed}.pdf")
        plt.show()


def plot_curv_est_res(res_folder, seeds):
    single_diff_list, cross_diff_list = [], []

    dist_axis = None
    for seed in seeds:
        res_folder_seed = os.path.join(res_folder, f"{seed}")

        filepath = os.path.join(res_folder_seed, f"curv_est_dist_w.npz")
        if os.path.exists(filepath):
            data = np.load(filepath)
            single_diff_list.append(data["single_diff"])
            cross_diff_list.append(data["cross_diff"])
            dist_axis = data['dist_list']
        else:
            print(f"!!! Missing results for dist_norm_w for seed {seed}")

    single_diff = np.concatenate(single_diff_list, axis=0)
    cross_diff = np.concatenate(cross_diff_list, axis=0)

    fig, ax1 = plt.subplots(1, 1, figsize=(3, 2), sharey=True)
    plot_curv_est_results(ax1, dist_axis, single_diff, cross_diff, "Norm of param distortion", "", set_ylabel=True)

    plt.tight_layout()
    plt.savefig(os.path.join(res_folder, "curv_est_test.pdf"), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=0, help='Number of random seeds to average over.')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--res_folder_root", type=str, default='results_curv',
                        help="Root folder to save results")
    parser.add_argument("--plot_res", action='store_true',
                        help="Flag to plot the final results")
    parser.add_argument("--plot_only", action='store_true',
                        help="Flag to plot the results without training for missing runs")

    args = parser.parse_args()
    res_folder_root = args.res_folder_root if args.res_folder_root else 'results_curv'
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

        for seed in seeds:
            print(f"\nRunning with seed {seed}")
            fix_seed(seed)

            res_folder_seed = os.path.join(res_folder, f"{seed}")
            if not os.path.exists(res_folder_seed):
                os.makedirs(res_folder_seed)

            run_curv_est()

    if args.plot_res:
        plot_curv_est_res(res_folder, seeds)
