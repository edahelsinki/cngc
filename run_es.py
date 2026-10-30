import os
import argparse
import numpy as np

# Set CUBLAS workspace config BEFORE importing torch
# This is critical for CUDA determinism
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from training.trainer_es import ModelTrainerWithEs

from utils.run_utils import get_seeds, fix_seed
from utils.plot_utils import plot_opt_res, plot_es_res

from dataset.data_utils import get_data_loaders, get_data
from models.model_utils import get_model


def get_filename(args):
    pc_id = "_pc" if args.use_pc else ""
    use_blocks = "_blocks" if args.use_blocks and args.use_pc else ""
    use_clean = "_clean" if args.use_clean_es and args.use_pc else ""
    use_val = "_val" if args.use_val else ""
    filename = f"results_{args.optimizer}{pc_id}{use_blocks}{use_clean}{use_val}.npz"
    return filename


def save_results(args, folder, metrics):
    filename = get_filename(args)
    res_path = os.path.join(folder, filename)
    np.savez(res_path, **metrics)
    print(f"Results saved to {res_path}")
    return


def run_training(args, folder, seed):
    print(f"\nTraining {args.model} for {args.dataset}")
    X, y, train_idx, task = get_data(args.dataset)
    y = y.reshape(-1, 1)

    test_ratio = 0.1
    dataloaders = get_data_loaders(X, y, args.batch_size, test_ratio, norm_X=args.norm_X, norm_y=False,
                                               task=task, train_idx=train_idx, use_val=args.use_val)

    # fix seeds
    fix_seed(seed)
    model = get_model(args.model, X.shape, y.shape, args.dataset, hidden_size=args.hidden_size)
    print(f"Number of parameters: {model.get_n_params()}")
    print(f"Training for {args.n_epochs} epochs with {args.optimizer} optimizer...")
    trainer = ModelTrainerWithEs(n_epochs=args.n_epochs,
                           lr=args.lr,
                           task=task,
                           use_pc=args.use_pc,
                           clean_updates=True,
                           opt=args.optimizer,
                           upd_every=1,
                           use_blocks=args.use_blocks,
                           noise_mult=args.noise_mult,
                           binary_noise=args.binary_noise,
                           subtract_clean=args.subtract_clean,
                           use_clean_es=args.use_clean_es
    )
    model, metrics = trainer.train(model, *dataloaders)

    # save results
    save_results(args, folder, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=0, help='Number of random seeds to average over.')
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument('--model', type=str, choices=['linear', 'mlp', 'cnn', 'transf'],
                        help='The type of the model.')
    parser.add_argument('--hidden_size', type=int, default=20, help='Size of MLP or CNN layers.')
    parser.add_argument('--dataset', type=str,
                        choices=['gen_reg', 'gen_class', 'bank', "income", "telescope", "occupancy", "air"],
                        help='The name of the dataset.')
    parser.add_argument('--optimizer', type=str, choices=['gd', 'adam', 'muon', 'sophia', 'sophiac'], default='adam',
                        help='The optimizer to use for training.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--n_epochs', type=int, default=60, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument("--use_blocks", action='store_true', help="Flag to use block structure in the preconditioner")
    parser.add_argument("--use_pc", action='store_true', help="Flag to use pre-conditioner in the optimization.")
    parser.add_argument("--noise_mult", type=float, default=0.0, help="Multiplier for the noise added to the targets.")
    parser.add_argument("--norm_X", action='store_true', help="Flag to divide X by std (mean is subtracted in any case).")
    parser.add_argument("--binary_noise", action='store_true', help="Flag to add binary noise to the targets instead of Gaussian.")
    parser.add_argument("--subtract_clean", action='store_true', help="Flag to subtract clean grad from noisy grad.")
    parser.add_argument("--use_clean_es", action='store_true',
                        help="Flag to use clean gradients for the early stopping stats.")
    parser.add_argument("--res_folder_root", type=str, default='results_es',
                        help="Root folder to save results")
    parser.add_argument("--plot_res", action='store_true',
                        help="Flag to plot the final results")
    parser.add_argument("--plot_only", action='store_true',
                        help="Flag to plot the results without training for missing runs")
    parser.add_argument("--use_val", action='store_true', help="Flag to use validation set for early stopping instead of credible value.")
    args = parser.parse_args()

    res_folder_root = args.res_folder_root if args.res_folder_root else 'results_es'
    model_task_folder = f"{args.model}_{args.dataset}_{args.optimizer}"

    res_folder = os.path.join(os.getcwd(), res_folder_root, model_task_folder)
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    if args.n_seeds > 0:
        seeds = get_seeds(args.n_seeds)
    else:
        seeds = [args.seed]
    print(f"Using seeds: {seeds}")

    if not args.plot_only:
        for seed in seeds:
            print(f"\nTraining with seed {seed}")
            # create a subfolder for the seed
            res_folder_seed = os.path.join(res_folder, f"{seed}")
            if not os.path.exists(res_folder_seed):
                os.makedirs(res_folder_seed)

            files = os.listdir(res_folder_seed)
            res_filename = get_filename(args)
            if res_filename not in files:
                run_training(args, folder=res_folder_seed, seed=seed)

    if args.plot_res:
        clean_id = "_clean" if args.use_clean_es else ""
        plot_es_res(res_folder, seeds, args.model, args.dataset, log_scale=False, show_legend=True, nsteps=None)
