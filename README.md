# Calibrated Target Noise Recovers Curvature from the Gradients
The repository contains the source code and scripts to reproduce the experiments. The provided SLURM scripts are the templates for the commands. The templates specify the required computational resources for the script execution.


## Dependencies
To reproduce the experiments, first install the dependencies by building a conda environment with `env.yml` file.


## Experiments
Below there are the descriptions of the commands to run the experiments, alongside with the scripts to schedule the jobs on a SLURM cluster, 
and the folders where the experiments are saved.

### 1. Study of the effects of centering, swamping, and noise design on curvature estimation.
The first three studies 1a -- 1c can run locally.
#### 1a. Curvature recovery with both centered and swamping estimators:
```
python run_curv_estimators.py --n_seeds 10 --plot_res
```
Results are saved in `results_curv`.

#### 1b. Bias of the swamping estimator and its dependence on the noise multiplier:
```
python run_est_bias.py --n_seeds 10 --plot_res
```
Results are saved in `results_bias`.

#### 1c. Noise design verification for the proposed estimator:
```
python run_noise_var.py --n_seeds 30 --plot_res
```
Results are saved in `results_gb_noise`.

#### 1d. The study of centering, swamping, and noise design on the eigenvalues of the true Hessian:
```
python run_binary_vs_gaussian.py --seed 42
```
Scripts to schedule the jobs: `scripts/run_noise.sh`
Results are saved in `results_noise_mult`.

### 2. Comparison of the GGN estimators with the baselines.
```
python compare_ngc.py --dataset air --model cnn --norm_X --hidden_size 20 --seed 42
```
Scripts to schedule the jobs: 
1. Comparison with the GGN: with an MLP `scripts/run_ngc_mlp.sh`, with a CNN `scripts/run_ngc_cnn.sh`. 
2. Comparison with the true Hessian: with an MLP `scripts/run_ngc_mlp_h_gt.sh`, with a CNN `scripts/run_ngc_cnn_h_gt.sh`.

Results are saved in `results_ngc_comparison`.


### 3. Optimization with CNGC preconditioning:

```
python run_opt.py --model transf --dataset air --optimizer gd --use_pc --use_blocks --warmup_epochs 0 --batch_size 128 --n_epochs 100 --lr 1e-4 --norm_X --noise_mult 1 --binary_noise --subtract_clean --seed 42
```

To reproduce our experiments on a SLURM cluster, run the script `scripts/submit_opt_all.sh` to schedule the runs for all studied combinations of models, dataset, and the optimizers with 10 random seeds.
The script to schedule an individual run can be found in `scripts/run_opt.sh`.The results are stored in `results_nn_opt`.


### 4. Early stoping with GGN estimator
```
python run_es.py --model cnn --dataset air --optimizer gd --use_pc --use_blocks --warmup_epochs 0 --batch_size 128 --n_epochs 100 --lr 1e-4 --norm_X --noise_mult 1 --binary_noise --subtract_clean --use_val --seed 42
```
For the SLURM cluster use the template in `scripts/submit_es_all_val` and `scripts/run_es.sh` for individual runs. The results are stored in `results_es`.

## Citation

```
@article{jamshidi2026calibrated,
  title={Calibrated Target Noise Recovers Curvature from the Gradients},
  author={Jamshidi, Arash and Haitsiukevich, Katsiaryna and Gionis, Aristides and Puolam{\"a}ki, Kai},
  journal={Advances in Neural Information Processing Systems},
  year={2026}
}
```