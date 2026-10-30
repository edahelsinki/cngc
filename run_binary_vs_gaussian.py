import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils.run_utils import fix_seed, get_seeds
from utils.matr_comparison_utils import test_noise
from utils.gen_data_utils import generate_data_nn
from utils.plot_utils import plot_noise_results
from utils.gen_data_utils import distort_model

from models.mlp import MLPRegressor


def run_dist_w(mlp, X, y,dist_norm_w_list, batch_size=128, n_repeats=10, class_task=True,
               filepath="results_noise/noise_dist_w.npz"):
    n_points = len(dist_norm_w_list)
    all_gauss = np.zeros((n_repeats, n_points))
    all_binary = np.zeros((n_repeats, n_points))
    all_binary_cl = np.zeros((n_repeats, n_points))
    all_binary_mult = np.zeros((n_repeats, n_points))

    for i, dist_norm_w in enumerate(dist_norm_w_list):
        model_dist = distort_model(mlp, dist_norm_w, X=X, y=y)
        for r in range(n_repeats):
            print(f"Evaluating: dist_norm_w {dist_norm_w}, repeat {r + 1}/{n_repeats}")
            eigv_diff_gauss, eigv_diff_bin, eigv_diff_bin_cl, eigv_diff_bin_mult = test_noise(
                model_dist, X, y, batch_size=batch_size,class_task=class_task, verbose=False)
            all_gauss[r, i] = eigv_diff_gauss
            all_binary[r, i] = eigv_diff_bin
            all_binary_cl[r, i] = eigv_diff_bin_cl
            all_binary_mult[r, i] = eigv_diff_bin_mult
            print()

    np.savez(filepath, g_dist=all_gauss, b_dist=all_binary, bc_dist=all_binary_cl, bm_dist=all_binary_mult,
             dist_list=dist_norm_w_list)
    return all_gauss, all_binary, all_binary_cl, all_binary_mult


def run_n_samples(model_dist, X_all, y_all, n_samples_list, batch_size=128, n_repeats=10, class_task=True,
                  filepath="results_noise/noise_n_samples.npz"):
    n_points = len(n_samples_list)
    all_gauss = np.zeros((n_repeats, n_points))
    all_binary = np.zeros((n_repeats, n_points))
    all_binary_cl = np.zeros((n_repeats, n_points))
    all_binary_mult = np.zeros((n_repeats, n_points))

    for i, n_samples in enumerate(n_samples_list):
        X = X_all[:n_samples]
        y = y_all[:n_samples]
        for r in range(n_repeats):
            print(f"Evaluating: n_samples {n_samples}, repeat {r + 1}/{n_repeats}")
            eigv_diff_gauss, eigv_diff_bin, eigv_diff_bin_cl, eigv_diff_bin_mult = test_noise(
                model_dist, X, y, batch_size=batch_size, class_task=class_task, verbose=False)
            all_gauss[r, i] = eigv_diff_gauss
            all_binary[r, i] = eigv_diff_bin
            all_binary_cl[r, i] = eigv_diff_bin_cl
            all_binary_mult[r, i] = eigv_diff_bin_mult
            print()

    np.savez(filepath, g_dist=all_gauss, b_dist=all_binary, bc_dist=all_binary_cl, bm_dist=all_binary_mult,
             samples_list=n_samples_list)
    return all_gauss, all_binary, all_binary_cl, all_binary_mult


def run_batch_size(model_dist, X, y, batch_list, n_repeats=10, class_task=True,
                   filepath="results_noise/noise_batch_size.npz"):
    n_points = len(batch_list)
    all_gauss = np.zeros((n_repeats, n_points))
    all_binary = np.zeros((n_repeats, n_points))
    all_binary_cl = np.zeros((n_repeats, n_points))
    all_binary_mult = np.zeros((n_repeats, n_points))

    for r in range(n_repeats):
        for i, batch_size in enumerate(batch_list):
            print(f"Evaluating: batch_size {batch_size}, repeat {r + 1}/{n_repeats}")
            eigv_diff_gauss, eigv_diff_bin, eigv_diff_bin_cl, eigv_diff_bin_mult = test_noise(
                model_dist, X, y, batch_size=batch_size, class_task=class_task, verbose=False)
            all_gauss[r, i] = eigv_diff_gauss
            all_binary[r, i] = eigv_diff_bin
            all_binary_cl[r, i] = eigv_diff_bin_cl
            all_binary_mult[r, i] = eigv_diff_bin_mult
            print()

    np.savez(filepath, g_dist=all_gauss, b_dist=all_binary, bc_dist=all_binary_cl, bm_dist=all_binary_mult,
             batch_list=batch_list)
    return all_gauss, all_binary, all_binary_cl, all_binary_mult


def plot_noise_res(res_folder, seeds, n_repeats):
    g_dist_list, b_dist_list, bc_dist_list, bm_dist_list = [], [], [], []
    g_samp_list, b_samp_list, bc_samp_list, bm_samp_list = [], [], [], []
    g_batch_list, b_batch_list, bc_batch_list, bm_batch_list = [], [], [], []

    dist_axis, samples_axis, batch_axis = None, None, None

    for seed in seeds:
        res_folder_seed = os.path.join(res_folder, f"{seed}")

        filepath = os.path.join(res_folder_seed, f"noise_dist_w_{n_repeats}.npz")
        if os.path.exists(filepath):
            data = np.load(filepath)
            g_dist_list.append(data['g_dist'])
            b_dist_list.append(data['b_dist'])
            bc_dist_list.append(data['bc_dist'])
            bm_dist_list.append(data['bm_dist'])
            dist_axis = data['dist_list']
        else:
            print(f"!!! Missing results for dist_norm_w for seed {seed}")

        filepath = os.path.join(res_folder_seed, f"noise_n_samples_{n_repeats}.npz")
        if os.path.exists(filepath):
            data = np.load(filepath)
            g_samp_list.append(data['g_dist'])
            b_samp_list.append(data['b_dist'])
            bc_samp_list.append(data['bc_dist'])
            bm_samp_list.append(data['bm_dist'])
            samples_axis = data['samples_list']
        else:
            print(f"!!! Missing results for n_samples for seed {seed}")

        filepath = os.path.join(res_folder_seed, f"noise_batch_size_{n_repeats}.npz")
        if os.path.exists(filepath):
            data = np.load(filepath)
            g_batch_list.append(data['g_dist'])
            b_batch_list.append(data['b_dist'])
            bc_batch_list.append(data['bc_dist'])
            bm_batch_list.append(data['bm_dist'])
            batch_axis = data['batch_list']
        else:
            print(f"!!! Missing results for batch_size for seed {seed}")

    g_dist = np.concatenate(g_dist_list, axis=0)
    b_dist = np.concatenate(b_dist_list, axis=0)
    bc_dist = np.concatenate(bc_dist_list, axis=0)
    bm_dist = np.concatenate(bm_dist_list, axis=0)

    g_samp = np.concatenate(g_samp_list, axis=0)
    b_samp = np.concatenate(b_samp_list, axis=0)
    bc_samp = np.concatenate(bc_samp_list, axis=0)
    bm_samp = np.concatenate(bm_samp_list, axis=0)

    g_batch = np.concatenate(g_batch_list, axis=0)
    b_batch = np.concatenate(b_batch_list, axis=0)
    bc_batch = np.concatenate(bc_batch_list, axis=0)
    bm_batch = np.concatenate(bm_batch_list, axis=0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.7, 2.1), sharey=False)
    plot_noise_results(ax1, dist_axis, g_dist, b_dist, bc_dist, bm_dist, "Norm of param distortion", "", y_max=3.1, set_ylabel=True)
    plot_noise_results(ax2, batch_axis, g_batch, b_batch, bc_batch, bm_batch, "Batch size", "", y_max=2.5, show_legend=False)
    plot_noise_results(ax3, samples_axis, g_samp, b_samp, bc_samp, bm_samp, "Number of samples", "", y_max=2.5, show_legend=True)

    plt.tight_layout()
    plt.savefig(os.path.join(res_folder, "combined_noise_experiments.pdf"), bbox_inches='tight')
    plt.show()


def run_seed():
    samples_list = [300, 500, 1000, 5000, 10000, 20000, 50000, 100000]

    # create a model, generate data, and distort the model once, to be used for all experiments
    mlp = MLPRegressor(in_features=n_features, out_features=1, hidden_layers=[5], add_bias=False, is_relu=True)
    X_all, y_all = generate_data_nn(mlp, n_samples=np.max(samples_list), n_features=n_features, noise_std=0.01,
                                    scale=1, multivar=True, class_task=class_task, seed=seed)

    model_dist = distort_model(mlp, dist_norm_w, X=X_all, y=y_all)
    filepath = f"{res_folder_seed}/noise_dist_w_{n_repeats}.npz"
    if os.path.exists(filepath):
        data = np.load(filepath)
        g_dist = data['g_dist']
        b_dist = data['b_dist']
        bc_dist = data['bc_dist']
        bm_dist = data['bm_dist']
        dist_list = data['dist_list']
    else:
        fix_seed(seed)
        dist_list = np.arange(0.1, 3.0, 0.5)
        g_dist, b_dist, bc_dist, bm_dist = run_dist_w(mlp, X_all[:n_samples], y_all[:n_samples], dist_list,
                                             batch_size=batch_size,
                                             n_repeats=n_repeats, class_task=class_task, filepath=filepath)

    filepath = f"{res_folder_seed}/noise_n_samples_{n_repeats}.npz"
    if os.path.exists(filepath):
        data = np.load(filepath)
        g_samp = data['g_dist']
        b_samp = data['b_dist']
        bc_samp = data['bc_dist']
        bm_samp = data['bm_dist']
        samples_list = data['samples_list']
    else:
        fix_seed(seed)
        g_samp, b_samp, bc_samp, bm_samp = run_n_samples(model_dist, X_all, y_all, samples_list, batch_size=batch_size,
                                                n_repeats=n_repeats, class_task=class_task, filepath=filepath)

    filepath = f"{res_folder_seed}/noise_batch_size_{n_repeats}.npz"
    if os.path.exists(filepath):
        data = np.load(filepath)
        g_batch = data['g_dist']
        b_batch = data['b_dist']
        bc_batch = data['bc_dist']
        bm_batch = data['bm_dist']
        batch_list = data['batch_list']
    else:
        fix_seed(seed)
        batch_list = [8, 16, 32, 64, 128, 256]
        g_batch, b_batch, bc_batch, bm_batch = run_batch_size(model_dist, X_all[:n_samples], y_all[:n_samples], batch_list,
                                                    n_repeats=n_repeats, class_task=class_task, filepath=filepath)

    if args.plot_res:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.5), sharey=True)
        plot_noise_results(ax1, dist_list, g_dist, b_dist, bc_dist, bm_dist, "Norm of param distortion", "", set_ylabel=True)
        plot_noise_results(ax2, samples_list, g_samp, b_samp, bc_samp, bm_samp, "Number of samples", "")
        plot_noise_results(ax3, batch_list, g_batch, b_batch, bc_batch, bm_batch, "Batch size", "", show_legend=True)

        plt.tight_layout()
        plt.savefig(f"{res_folder_seed}/combined_noise_experiments_{seed}.pdf")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=0, help='Number of random seeds to average over.')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--res_folder_root", type=str, default='results_noise_mult',
                        help="Root folder to save results")
    parser.add_argument("--plot_res", action='store_true',
                        help="Flag to plot the final results")
    parser.add_argument("--plot_only", action='store_true',
                        help="Flag to plot the results without training for missing runs")

    args = parser.parse_args()
    res_folder_root = args.res_folder_root if args.res_folder_root else 'results_noise_mult'
    res_folder = os.path.join(os.getcwd(), res_folder_root)
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    if args.n_seeds > 0:
        seeds = get_seeds(args.n_seeds)
    else:
        seeds = [args.seed]
    print(f"Using seeds: {seeds}")
    n_repeats = 1

    if not args.plot_only:
        n_features = 20
        batch_size = 64
        class_task = False
        dist_norm_w = 1.1
        n_samples = 50000

        for seed in seeds:
            print(f"\nRunning with seed {seed}")
            fix_seed(seed)

            res_folder_seed = os.path.join(res_folder, f"{seed}")
            if not os.path.exists(res_folder_seed):
                os.makedirs(res_folder_seed)

            run_seed()

    if args.plot_res:
        plot_noise_res(res_folder, seeds, n_repeats)
