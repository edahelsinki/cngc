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
from utils.estimators import get_ggn_batch, get_ngc
from utils.plot_utils import plot_est_bias_results
from utils.matr_comparison_utils import get_matr_diff


def run_bias(model_dist, X, y, mult_list, batch_size=128, class_task=True, n_repeats=50,
               filepath="results_bias/est_bias.npz"):
    X = torch.tensor(X).float()
    y = torch.tensor(y[:, None]).float()
    dataset = TensorDataset(X, y)
    dataloader = get_dataload_with_seed(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    n_points = len(mult_list)
    bias_array = np.zeros((1, n_points))

    for i, mult in enumerate(mult_list):
        print(f"Evaluating: mult =  {mult}")
        # get GGN
        ggn = get_ggn_batch(model_dist, dataloader, class_task=class_task)

        cross_batch_cum = torch.zeros_like(ggn)
        for _ in range(n_repeats):
            # GGN est is "uncentered" since the clean grad is  not subtracted, so mean is calculated across batches
            cross_batch = get_ngc(model_dist, dataloader, class_task=class_task, noise_mult=mult, subtract_clean=False)
            cross_batch_cum += cross_batch

        cross_batch_mean = cross_batch_cum / n_repeats
        bias = get_matr_diff(ggn, cross_batch_mean)
        bias_array[0, i] = bias

    np.savez(filepath, bias_diff=bias_array, mult_list=mult_list)
    return bias_array


def run_est_bias(n_repeats):
    # create a model, generate data, and distort the model once, to be used for all experiments
    mlp = MLPRegressor(in_features=n_features, out_features=1, hidden_layers=[5], add_bias=False, is_relu=True)
    X_all, y_all = generate_data_nn(mlp, n_samples=n_samples, n_features=n_features, noise_std=0.01,
                                    scale=1, multivar=True, class_task=class_task, seed=seed)
    model_dist = distort_model(mlp, dist_norm_w=2.6, X=X_all, y=y_all)

    filepath = f"{res_folder_seed}/est_bias.npz"
    if os.path.exists(filepath):
        data = np.load(filepath)
        bias_array = data["bias_diff"]
        mult_list = data['mult_list']
    else:
        fix_seed(seed)
        mult_list = np.array([0.1, 1.0, 2.0, 5.0, 10.0])
        bias_array = run_bias(model_dist, X_all, y_all, mult_list, batch_size=batch_size, class_task=class_task,
                              n_repeats=n_repeats, filepath=filepath)

    if args.plot_res:
        fig, ax1 = plt.subplots(1, 1, figsize=(3.15, 2), sharey=True)
        plot_est_bias_results(ax1, mult_list, bias_array, "Noise multiplier", "", set_ylabel=True)

        plt.tight_layout()
        plt.savefig(f"{res_folder_seed}/est_bias_test_{seed}.pdf")
        plt.show()


def plot_est_bias_res(res_folder, seeds):
    bias_list = []

    mult_axis = None
    for seed in seeds:
        res_folder_seed = os.path.join(res_folder, f"{seed}")

        filepath = os.path.join(res_folder_seed, f"est_bias.npz")
        if os.path.exists(filepath):
            data = np.load(filepath)
            bias_list.append(data["bias_diff"])
            mult_axis = data['mult_list']
        else:
            print(f"!!! Missing results for seed {seed}")

    bias_diff = np.concatenate(bias_list, axis=0)

    fig, ax1 = plt.subplots(1, 1, figsize=(3.15, 2), sharey=True)
    plot_est_bias_results(ax1, mult_axis, bias_diff, "Noise multiplier", "", set_ylabel=True)

    plt.tight_layout()
    plt.savefig(os.path.join(res_folder, "est_bias_test.pdf"), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=0, help='Number of random seeds to average over.')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--res_folder_root", type=str, default='results_bias',
                        help="Root folder to save results")
    parser.add_argument("--plot_res", action='store_true',
                        help="Flag to plot the final results")
    parser.add_argument("--plot_only", action='store_true',
                        help="Flag to plot the results without training for missing runs")

    args = parser.parse_args()
    res_folder_root = args.res_folder_root if args.res_folder_root else 'results_bias'
    res_folder = os.path.join(os.getcwd(), res_folder_root)
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    if args.n_seeds > 0:
        seeds = get_seeds(args.n_seeds)
    else:
        seeds = [args.seed]
    print(f"Using seeds: {seeds}")
    n_repeats = 50

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

            run_est_bias(n_repeats)

    if args.plot_res:
        plot_est_bias_res(res_folder, seeds)
