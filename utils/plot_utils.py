import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

custom_dash_patterns = [
    # (0, ()),                   # 1. Solid
    (0, (5, 1)),               # 2. Dense Dash
    (0, (5, 5)),               # 3. Loose Dash
    (0, (1, 1)),               # 4. Dense Dot
    (0, (1, 5)),               # 5. Loose Dot
    (0, (10, 2, 1, 2)),        # 6. Long Dash-Dot
    (0, (5, 2, 1, 2, 1, 2)),   # 7. Double Dash-Dot
    (0, (15, 3, 5, 3))         # 8. Long Dash with Varying Gaps
]

colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']


def plot_opt_res(res_folder, seeds, model_name, dataset_name, log_scale=False, show_legend=True, nsteps=None,
                 add_es=False, clean_id="", only_gd_pc=False):
    fig, axs = plt.subplots(figsize=(3, 5), nrows=2, ncols=1, sharex=True)

    styles = ['-',  (0, (1, 5)), '--', '-.', ':', (0, (5, 1)), (0, (5, 5)), (0, (1, 1)), (0, (1, 5))]
    colors = ['tab:blue', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:blue']
    st_i = 0

    optimizers = ['gd', 'adam', 'muon', 'sophia', 'sophiac']
    for opt in optimizers:
        for use_pc in [True, False]:
            pc_id = "_pc" if use_pc else ""

            if only_gd_pc:
                if opt != 'gd' or not use_pc:
                    continue  # only plot GD with preconditioner

            if opt != 'gd' and use_pc:
                continue # preconditioning is applied only for GD

            for use_blocks in [False, True]:
                if not use_pc and use_blocks:
                    continue  # blocks only make sense when using preconditioner

                if opt == "gd" and use_pc and not use_blocks:
                    continue  # the preconditioner is always applied with blocks

                blocks_id = "_blocks" if use_blocks else ""

                for use_ggn in [False, True]:
                    if (opt != 'gd' or not use_pc) and use_ggn:
                        continue  # GGN is applied only for GD

                    ggn_id = "_ggn" if use_ggn else ""
                    filename = f"results_{opt}{pc_id}{blocks_id}{ggn_id}{clean_id}.npz"

                    train_losses, test_losses = [], []
                    es_stats = []
                    for seed in seeds:
                        res_path = os.path.join(res_folder, f"{seed}", filename)
                        if os.path.exists(res_path):
                            data = np.load(res_path)
                            train_losses.append(data['train_losses'])
                            test_losses.append(data['test_losses'])

                            if add_es and 'es_stats' in data:
                                es_stats.append(data['es_stats'])

                    train_losses = np.array(train_losses)
                    test_losses = np.array(test_losses)
                    label = get_label(opt, pc_id, blocks_id, ggn_id)

                    if len(train_losses) < len(seeds) or len(test_losses) < len(seeds):
                        print(f"!!! Missing results for {label} for some seeds."
                              f" Averaging over {len(train_losses)} seeds instead of {len(seeds)}.")

                    if len(train_losses) > 0 and len(test_losses) > 0:

                        if nsteps is not None and train_losses.shape[1] > nsteps:
                            # option to plot less steps to make the plots more informative
                            train_losses = train_losses[:, :nsteps]
                            test_losses = test_losses[:, :nsteps]

                        epochs = np.arange(len(train_losses[0])) + 1

                        train_mean = np.mean(train_losses, axis=0)
                        test_mean = np.mean(test_losses, axis=0)
                        axs[0].plot(epochs, train_mean, label=label, linestyle=styles[st_i], color=colors[st_i])
                        axs[1].plot(epochs, test_mean, label=label, linestyle=styles[st_i], color=colors[st_i])

                        # plot standard deviation as shaded area
                        train_std = np.std(train_losses, axis=0)
                        test_std = np.std(test_losses, axis=0)
                        axs[0].fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2, color=colors[st_i])
                        axs[1].fill_between(epochs, test_mean - test_std, test_mean + test_std, alpha=0.2, color=colors[st_i])

                        if len(es_stats) > 0:
                            es_stats = np.array(es_stats)

                            # Find the first increase in the statistics
                            diff = np.diff(es_stats, axis=1)
                            mask = diff > 0  # True where value increases
                            has_increase = mask.any(axis=1)  # (10,) bool
                            stopping_points = np.where(has_increase, np.argmax(mask, axis=1), -1)
                            p = np.median(stopping_points)
                            p_std = np.std(stopping_points)
                            if clean_id != "":
                                label_es = "GradStop"
                                color_es = "blue"
                            else:
                                label_es = "Stop. point (ours)"
                                color_es = 'red'
                            axs[1].axvline(x=p, color=color_es, linestyle='--', linewidth=1.5, label=label_es, markersize=4)
                            axs[1].axvspan(np.maximum(p - p_std, 0), p + p_std, alpha=0.2, color=color_es)
                            plt.legend()

                        if log_scale:
                            axs[0].set_yscale('log', base=2)
                            axs[1].set_yscale('log', base=2)

                    st_i += 1  ## keep the counter for styles

    axs[1].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Test Loss")
    if show_legend:
        axs[1].legend(fontsize=9, loc='upper right')

    fig.suptitle(f"{model_name} on {dataset_name}")
    fig.tight_layout()

    if os.path.exists(res_folder):
        fig.savefig(os.path.join(res_folder, f"opt_{model_name}_{dataset_name}{clean_id}.pdf"))
    fig.show()
    return


def get_test_loss_th(res_folder, seeds, th=0.9, filename="results_gd_pc_blocks.npz"):
    test_losses = []
    for seed in seeds:
        res_path = os.path.join(res_folder, f"{seed}", filename)
        if os.path.exists(res_path):
            data = np.load(res_path)
            test_losses.append(data['test_losses'])

    test_losses = np.array(test_losses)
    test_losses_mean = np.mean(test_losses, axis=0)

    # find index where the test loss is less than 0.9 of the minimal loss
    th = np.max(test_losses_mean) - th * (np.max(test_losses_mean) - np.min(test_losses_mean))
    return th


def build_walltime_table_md(res_folder, seeds, res_dict, get_full_time=True, res_folder_es=None, use_es=False):
    test_th = get_test_loss_th(res_folder, seeds)
    print(f"Test loss threshold is {test_th:.4f}")
    for opt_full in res_dict.keys():
        filename = f"results_{opt_full}.npz"

        walltimes = []
        test_losses = []
        for seed in seeds:
            res_path = os.path.join(res_folder, f"{seed}", filename)
            if os.path.exists(res_path):
                data = np.load(res_path)
                walltimes.append(data['wall_time'])
                test_losses.append(data['test_losses'])

        walltimes = np.array(walltimes)  # dim = (n_seeds, n_epochs)
        test_losses = np.array(test_losses)

        if len(walltimes) < len(seeds) or len(test_losses) < len(seeds):
            print(f"!!! Missing results for {opt_full} for some seeds."
                  f" Averaging over {len(walltimes)} seeds instead of {len(seeds)}.")

        if len(walltimes) > 0 and len(test_losses) > 0:

            if get_full_time or res_folder_es is None:
                walltimes_mean = np.mean(np.mean(walltimes, axis=1), axis=0)
                walltimes_std = np.std(np.mean(walltimes, axis=1), axis=0)
                res_dict[opt_full].append(f"{walltimes_mean:.2f} ± {walltimes_std:.2f}")
            elif use_es is True:
                if opt_full == "gd_pc_blocks":
                    # load early stopping times
                    es_stats = []
                    for seed in seeds:
                        res_path = os.path.join(res_folder_es, f"{seed}", filename)
                        if os.path.exists(res_path):
                            data = np.load(res_path)
                            es_stats.append(data['es_stats'])

                    es_stats = np.array(es_stats)

                    # Find the first increase in the statistics
                    diff = np.diff(es_stats, axis=1)
                    mask = diff > 0  # True where value increases
                    has_increase = mask.any(axis=1)  # (10,) bool
                    stopping_points = np.where(has_increase, np.argmax(mask, axis=1), -1)

                    walltimes_converg = []
                    for i, p_i in enumerate(stopping_points):
                        if p_i < 0:
                            print(f"Warning: No stopping point found for seed = {seeds[i]}. Using full training time.")
                            wall_time_curr = np.sum(walltimes, axis=1)
                        else:
                            wall_time_curr = np.sum(walltimes[i, :p_i + 1])

                        walltimes_converg.append(wall_time_curr)
                    walltimes_converg = np.array(walltimes_converg)
                else:
                    walltimes_converg = np.sum(walltimes, axis=1)
                walltimes_converg_mean = np.mean(walltimes_converg, axis=0)
                walltimes_converg_std = np.std(walltimes_converg, axis=0)
                res_dict[opt_full].append(f"{walltimes_converg_mean:.2f} ± {walltimes_converg_std:.2f}")
            else:
                # find the total time it takes for the optimizer to reach 0.9 of test loss of GD with PC and blocks or until the end of the training
                test_losses_mean = np.mean(test_losses, axis=0)
                idx = np.where(test_losses_mean <= test_th)[0]
                if len(idx) > 0:
                    idx = idx[0]  # take the first time where 90% of the convergence were reached
                    print(f"Method {opt_full} reaches test threshold in step {idx}.")
                    walltimes = walltimes[:, :idx]
                    not_cov_id = ""
                else:
                    print(f"The method {opt_full} didn't converge")
                    not_cov_id = "^"

                walltimes_converg = np.sum(walltimes, axis=1)

                walltimes_converg_mean = np.mean(walltimes_converg, axis=0)
                walltimes_converg_std = np.std(walltimes_converg, axis=0)
                res_dict[opt_full].append(f"{walltimes_converg_mean:.2f} ± {walltimes_converg_std:.2f}{not_cov_id}")
    return res_dict


def build_abl_table_md(res_folder, seeds, res_dict, test_loss=True):
    test_th = get_test_loss_th(res_folder, seeds, th=0.9, filename="results_gd_pc_blocks_ggn.npz")
    # print(f"Test loss threshold is {test_th:.4f}")
    for opt_full in res_dict.keys():
        filename = f"results_{opt_full}.npz"

        test_losses = []
        for seed in seeds:
            res_path = os.path.join(res_folder, f"{seed}", filename)
            if os.path.exists(res_path):
                data = np.load(res_path)
                test_losses.append(data['test_losses'])

        test_losses = np.array(test_losses)
        if len(test_losses) < len(seeds):
            print(f"!!! Missing results for {opt_full} for some seeds."
                  f" Averaging over {len(test_losses)} seeds instead of {len(seeds)}.")

        if len(test_losses) > 0:

            if test_loss:
                test_losses_last = np.min(test_losses, axis=1)  # take the last test loss for each seed
                test_losses_mean = np.mean(test_losses_last, axis=0)
                test_losses_std = np.std(test_losses_last, axis=0)
                res_dict[opt_full].append(f"{test_losses_mean:.4f} ± {test_losses_std:.4f}")
                print(f"{test_losses_mean:.4f} ± {test_losses_std:.4f}")
            else:
                # find the number of iterations it takes for the optimizer to reach 0.9 of test loss of GD with PC
                # and blocks or until the end of the training
                not_cov_id = ""
                indices = []
                for s in range(len(test_losses)):
                    test_losses_curr = test_losses[s, :]

                    idx = np.where(test_losses_curr <= test_th)[0]
                    if len(idx) > 0:
                        idx = idx[0]  # take the first time where 90% of the convergence were reached
                        # print(f"Method {opt_full} reaches test threshold in step {idx}.")
                    else:
                        print(f"The method {opt_full} didn't converge for seed {seeds[s]}. Using full training time.")
                        idx = len(test_losses_curr)
                        not_cov_id = "^"
                    indices.append(idx)

                indices = np.array(indices)
                indices_mean = np.mean(indices, axis=0)
                indices_std = np.std(indices, axis=0)
                res_dict[opt_full].append(f"{indices_mean:.2f} ± {indices_std:.2f}{not_cov_id}")
        else:
            res_dict[opt_full].append("N/A")
    return res_dict


def plot_es_res(res_folder, seeds, model_name, dataset_name, log_scale=False, show_legend=True, nsteps=None):
    fig, axs = plt.subplots(figsize=(3, 2.8), nrows=1, ncols=1, sharex=True)

    styles = ['-', ]
    colors = ['tab:blue']
    st_i = 0

    filename = f"results_gd_pc_blocks.npz"
    filename_clean = f"results_gd_pc_blocks_clean.npz"
    filename_val_set = f"results_gd_pc_blocks_val.npz"

    es_stats, test_losses, _ = load_es_stats(res_folder, filename, seeds)
    es_stats_clean, test_losses_clean, _ = load_es_stats(res_folder, filename_clean, seeds)
    es_stats_val, test_losses_val, _ = load_es_stats(res_folder, filename_val_set, seeds)

    if len(test_losses) > 0:
        if nsteps is not None and test_losses.shape[1] > nsteps:
            # option to plot less steps to make the plots more informative
            test_losses = test_losses[:, :nsteps]
            es_stats = es_stats[:, :nsteps]
            es_stats_clean = es_stats_clean[:, :nsteps]
            es_stats_val = es_stats_val[:, :nsteps]

        epochs = np.arange(len(test_losses[0])) + 1
        test_mean = np.mean(test_losses, axis=0)
        axs.plot(epochs, test_mean, label="GD, PC (ours)", linestyle=styles[st_i], color=colors[st_i])

        # plot standard deviation as shaded area
        test_std = np.std(test_losses, axis=0)
        axs.fill_between(epochs, test_mean - test_std, test_mean + test_std, alpha=0.2, color=colors[st_i])

        # Find the first increase in the statistics
        if len(es_stats) > 0:
            points, p, p_std = find_stop_points(es_stats)
            test_loss = np.mean(test_losses, axis=0)[int(p)]
            print(f"test loss at stopping point (ours) = {test_loss:.4f}")
            axs.axvline(x=p, color='red', linestyle='--', linewidth=1.5, label="Stop. point (ours)", markersize=4)
            axs.axvspan(np.maximum(p - p_std, 0), p + p_std, alpha=0.12, color='red')

        if len(es_stats_clean) > 0:
            points_clean, p_clean, p_std_clean = find_stop_points(es_stats_clean)
            test_loss = np.mean(test_losses_clean, axis=0)[int(p_clean)]
            print(f"test loss at stopping point (clean) = {test_loss:.4f}")
            axs.axvline(x=p_clean, color='blue', linestyle=':', linewidth=1.5, label="GradStop", markersize=4)
            axs.axvspan(np.maximum(p_clean - p_std_clean, 0), p_clean + p_std_clean, alpha=0.12, color='blue')

        if len(es_stats_val) > 0:
            points_val, p_val, p_std_val = find_stop_points(es_stats_val)
            test_loss = np.mean(test_losses_val, axis=0)[int(p_val)]
            print(f"test loss at stopping point (val) = {test_loss:.4f}")
            axs.axvline(x=p_val, color='green', linestyle='-.', linewidth=1.5, label="Stop. point (val)", markersize=4)
            axs.axvspan(np.maximum(p_val - p_std_val, 0), p_val + p_std_val, alpha=0.12, color='green')

        if log_scale:
            axs.set_yscale('log', base=2)

    axs.set_xlabel("Epoch")
    axs.set_ylabel("Test Loss")
    if show_legend:
        axs.legend(fontsize=9, loc='upper right')

    fig.suptitle(f"{model_name} on {dataset_name}")
    fig.tight_layout()

    if os.path.exists(res_folder):
        fig.savefig(os.path.join(res_folder, f"es_{model_name}_{dataset_name}.pdf"))
    fig.show()
    return


def build_table_es_res(res_folder, seeds, res_dict, patience=1):
    for res_name in res_dict.keys():
        if res_name == 'ours':
            filename = f"results_gd_pc_blocks.npz"
        else:
            filename = f"results_gd_pc_blocks_{res_name}.npz"

        es_stats, test_losses, _ = load_es_stats(res_folder, filename, seeds)
        if len(test_losses) > 0:
            points, _, _ = find_stop_points(es_stats, patience=patience)
            selected_test_losses = []
            for i, p in enumerate(points):
                selected_test_losses.append(test_losses[i, p])

            selected_test_losses = np.array(selected_test_losses)
            # print(selected_test_losses)
            test_mean = np.mean(selected_test_losses)
            test_std = np.std(selected_test_losses)
            res_dict[res_name].append(f"{test_mean:.4f} ± {test_std:.4f}")
        else:
            res_dict[res_name].append("N/A")

    return res_dict


def find_stop_points(es_stats, patience=3):
    diff = np.diff(es_stats, axis=1)
    mask = diff > 0  # True where value increases, shape (n_seeds, n_epochs-1)

    cum_increases = np.cumsum(mask, axis=1)  # running count of increases up to and including position j
    reached = cum_increases >= patience
    has_enough = reached.any(axis=1)

    j = np.argmax(reached, axis=1)  # first index where cumulative count hits `patience`

    stopping_points = np.where(has_enough, j, es_stats.shape[1] - 1)

    p = np.median(stopping_points)
    p_std = np.std(stopping_points)
    return stopping_points, p, p_std


def load_es_stats(res_folder, filename, seeds):
    train_losses, test_losses = [], []
    es_stats = []
    for seed in seeds:
        res_path = os.path.join(res_folder, f"{seed}", filename)
        if os.path.exists(res_path):
            data = np.load(res_path)
            train_losses.append(data['train_losses'])
            test_losses.append(data['test_losses'])
            es_stats.append(data['es_stats'])

    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    es_stats = np.array(es_stats)

    if len(train_losses) < len(seeds) or len(test_losses) < len(seeds) or len(es_stats) < len(seeds):
        print(f"!!! Missing results for early stopping {filename} for some seeds."
              f" Averaging over {len(train_losses)} seeds instead of {len(seeds)}.")
    return es_stats, test_losses, train_losses


def get_label(opt: str, pc_id: str, blocks_id: str, ggn_id:str="") -> str:
    label = str(opt).capitalize() if opt != "gd" else "GD"
    if opt == "sophia":
        label = "SophiaG"
    elif opt == "sophiac":
        label = "SophiaC (ours)"

    if len(pc_id) > 0:
        label += ", PC (ours)"
        # if len(blocks_id) > 0:
        #     label += ", bl"

    if len(ggn_id) > 0:
        label += ", GGN"
    return label


def plot_noise_results(ax, x_axis, all_gauss, all_binary, all_binary_cl, all_binary_mult,
                       x_label, title,
                       y_max=3.1, set_ylabel=False, show_legend=False):
    mean_g, std_g = np.mean(all_gauss, axis=0), np.std(all_gauss, axis=0)
    mean_b, std_b = np.mean(all_binary, axis=0), np.std(all_binary, axis=0)
    mean_bc, std_bc = np.mean(all_binary_cl, axis=0), np.std(all_binary_cl, axis=0)
    mean_bm, std_bm = np.mean(all_binary_mult, axis=0), np.std(all_binary_mult, axis=0)
    cap = y_max
    # ax.plot(x_axis, mean_g, label="Gaussian", color='tab:green', lw=2, linestyle='-.',
    #         marker='d', markersize=6)
    # ax.fill_between(x_axis, np.maximum(mean_g - std_g, 0.), np.minimum(mean_g + std_g, cap), color='tab:green', alpha=0.2, linestyle='-.')

    ax.plot(x_axis, mean_b, label="Swamping", color='tab:orange', lw=2, linestyle='--',
            marker='d', markersize=4)
    ax.fill_between(x_axis, np.maximum(mean_b - std_b, 0.), np.minimum(mean_b + std_b, cap), color='tab:orange', alpha=0.2, linestyle='-')

    ax.plot(x_axis, mean_bc, label="Centered", color='tab:blue', lw=2, linestyle='-',
            marker='d', markersize=6)
    ax.fill_between(x_axis, np.maximum(mean_bc - std_bc, 0), np.minimum(mean_bc + std_bc, cap), color='tab:blue', alpha=0.2, linestyle='-')

    ax.plot(x_axis, mean_bm, label="Swamping, mult. 10", color='tab:red', lw=2, linestyle=(0, (1, 2)),
            marker='d', markersize=4)
    ax.fill_between(x_axis, np.maximum(mean_bm - std_bm, 0.), np.minimum(mean_bm + std_bm, cap), color='tab:red',
                    alpha=0.2, linestyle='-.')

    if x_label.lower() == "batch size":
        ax.set_xscale('log', base=2)

    if x_label.lower() == "number of samples":
        ax.set_xscale('log', base=10)

    ax.set_ylim(0, cap)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    if set_ylabel:
        ax.set_ylabel("Hessian \nEigen val. diff.")

    if show_legend:
        ax.legend()


def plot_ngc_comparison(res, n_batches, batch_size, res_folder, log_scale=False, use_h=False, title=None):
    x_labels = np.array(n_batches) * batch_size
    is_time_plot = "time" in title
    model_dataset = res_folder.split("/")[-1]
    model_name = model_dataset.split("_")[0]
    dataset_name = model_dataset.split("_")[1]

    size = (4, 2) if is_time_plot else (2.9, 2)
    if use_h and "eigv" in title:
        size = (3.15, 2)
    fig, ax = plt.subplots(figsize=size)
    for i, est in enumerate(res.keys()):
        if not is_time_plot and est == "ggn":
            continue  ## no difference for GGN matrix, because GGN is the ground truth

        est_name = est.upper() if est != "hutch" else "Hutch."
        if dataset_name != "air" and est_name == "GNB_MSE":
            continue

        res_i = np.array(res[est])
        mean = np.mean(res_i, axis=1)
        std = np.std(res_i, axis=1)

        if est_name == "NGC":
            est_name = "CNGC (ours)"  # the method is names as "NGC" in the code
        elif est_name == "GNB_MSE":
            est_name = "GNB (MSE)"

        if est == "ngc":
            color = 'tab:blue'
            linestyle = '-'
        else:
            color = colors[i]
            linestyle = custom_dash_patterns[i]

        ax.plot(x_labels, mean, label=est_name, color=color, linestyle=linestyle, marker='d', markersize=6)
        ax.fill_between(x_labels, np.maximum(mean - std, 0), mean + std, alpha=0.2, color=color, linestyle=linestyle)

    ax.set_xlabel("# samples")
    if log_scale:
        ax.set_yscale('log', base=10)

    if is_time_plot:
        ax.set_ylabel("Time (s)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    elif "eigv" in title:
        if use_h:
            ylabel_title = "Hessian \nEigen val. diff."
        else:
            ylabel_title = "Eigen val. diff."
        ax.set_ylabel(ylabel_title)
    else:
        if use_h:
            ylabel_title = "Hessian diff."
        else:
            ylabel_title = "GGN diff."
        ax.set_ylabel(ylabel_title)

    fig.suptitle(f"{model_name} on {dataset_name}")

    fig.set_layout_engine('constrained')
    fig.savefig(f'{res_folder}/{model_dataset}_{title}.pdf')


def plot_curv_est_results(ax, x_axis, single_diff, cross_diff, x_label, title, set_ylabel=False):
    mean_single, std_single = np.mean(single_diff, axis=0), np.std(single_diff, axis=0)
    mean_cross, std_cross = np.mean(cross_diff, axis=0), np.std(cross_diff, axis=0)
    cap = 10.

    ax.plot(x_axis, mean_cross, label="Swamping", color='tab:orange', lw=2, linestyle='--',
            marker='d', markersize=4)
    ax.fill_between(x_axis,
                    np.maximum(mean_cross - std_cross, 0),
                    np.minimum(mean_cross + std_cross, cap), color='tab:orange',
                    alpha=0.2, linestyle='-')

    ax.plot(x_axis, mean_single, label="Centered", color='tab:blue', lw=2, linestyle='-',
            marker='d', markersize=6)
    ax.fill_between(x_axis,
                    np.maximum(mean_single - std_single, 0.),
                    np.minimum(mean_single + std_single, cap), color='tab:blue',
                    alpha=0.2, linestyle='-')


    ax.set_title(title)
    ax.set_xlabel(x_label)

    if set_ylabel:
        ax.set_ylabel("Relative diff.")

    ax.legend()


def plot_est_bias_results(ax, x_axis, bias_array, x_label, title, set_ylabel=False):
    mean_bias = np.mean(bias_array, axis=0)
    std_bias = np.std(bias_array, axis=0)

    ax.plot(x_axis, mean_bias, label="Swamping", color='tab:orange', lw=2, linestyle='--',
            marker='d', markersize=6)

    ax.fill_between(x_axis,
                    np.maximum(mean_bias - std_bias, 0),
                    mean_bias + std_bias, color='tab:orange',
                    alpha=0.2, linestyle='-')

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_title(title)
    ax.set_xlabel(x_label)

    if set_ylabel:
        ax.set_ylabel("Relative diff.")

    ax.legend()


def plot_gb_noise_results(ax, x_axis, binary_diff, gaussian_diff, x_label, title, set_ylabel=False):
    mean_binary, std_binary = np.mean(binary_diff, axis=0), np.std(binary_diff, axis=0)
    mean_gaussian, std_gaussian = np.mean(gaussian_diff, axis=0), np.std(gaussian_diff, axis=0)
    cap = 10.

    ax.plot(x_axis, mean_gaussian, label="Gaussian", color='tab:pink', lw=2, linestyle=':',
            marker='d', markersize=4)
    ax.fill_between(x_axis,
                    np.maximum(mean_gaussian - std_gaussian, 0),
                    np.minimum(mean_gaussian + std_gaussian, cap), color='tab:pink',
                    alpha=0.2, linestyle='-')

    ax.plot(x_axis, mean_binary, label="Rademacher", color='tab:blue', lw=2, linestyle='-',
            marker='d', markersize=6)
    ax.fill_between(x_axis,
                    np.maximum(mean_binary - std_binary, 0.),
                    np.minimum(mean_binary + std_binary, cap), color='tab:blue',
                    alpha=0.2, linestyle='-')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylim(0.03, 0.17)

    if set_ylabel:
        ax.set_ylabel("Relative diff.")

    ax.legend()
