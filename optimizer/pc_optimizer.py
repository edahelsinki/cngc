import torch
from enum import Enum


class GradientMeanEstimator():
    """
    Accumulates gradients and tracks their running mean.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._n     = 0
        self._mean  = None

    def update(self, grad: torch.Tensor) -> None:
        """Add one gradient sample (1-D vector)."""
        self._n += 1
        if self._mean is None:
            self._mean = grad.clone()
        else:
            self._mean += (grad - self._mean) / self._n

    def finalise(self) -> torch.Tensor:
        """Return the mean and reset for the next window."""
        if self._n < 1:
            raise RuntimeError("No gradient samples accumulated.")
        mean = self._mean
        self.reset()
        return mean


class GradientCovarianceEstimator():
    """
    Accumulates noisy gradients and computes their covariance on demand (without storing full grad history

    If model_params are provided, computes a block-diagonal covariance where
    each block corresponds to one layer.
    """

    def __init__(self, param_sizes=None):
        """
        Args:
            param_sizes: if provided only blocks of the covariance are calculated
        """
        self.param_sizes = param_sizes
        self.reset()

    def reset(self):
        self._n      = 0
        self._mean   = None
        self._M2     = None   # either one (n_params, n_params) tensor, # or a list of (block_size, block_size) tensors
        self.cov     = None

    def update(self, grad: torch.Tensor) -> None:
        """Add one gradient sample (1-D vector)."""
        self._n += 1
        if self._mean is None:
            self._mean = grad.clone()
            if self.param_sizes is not None:
                self._M2 = [
                    torch.zeros(s, s, device=grad.device)
                    for s in self.param_sizes
                ]
            else:
                self._M2 = torch.zeros(len(grad), len(grad), device=grad.device)
        else:
            delta      = grad - self._mean
            self._mean += delta / self._n
            delta2     = grad - self._mean

            if self.param_sizes is not None:
                for block, (g1, g2) in zip(
                    self._M2, self._iter_blocks(delta, delta2)
                ):
                    block += torch.outer(g1, g2)
            else:
                self._M2 += torch.outer(delta, delta2)

    def finalise(self) -> torch.Tensor | list[torch.Tensor]:
        """
        Compute and cache the covariance from accumulated samples.

        Returns:
            Full mode:  a single (n_params, n_params) covariance tensor.
            Block mode: a list of per-layer (block_size, block_size) tensors.
        """
        if self._n < 2:
            raise RuntimeError("Need at least 2 gradient samples to compute covariance.")
        if self.param_sizes is not None:
            self.cov = [M2 / (self._n - 1) for M2 in self._M2]
        else:
            self.cov = self._M2 / (self._n - 1)

        cov = self.cov

        self.reset()
        return cov

    def _iter_blocks(
        self, *vecs: torch.Tensor
    ) -> list[tuple[torch.Tensor, ...]]:
        """Split one or more flat vectors into per-layer blocks."""
        result, offset = [], 0
        for size in self.param_sizes:
            result.append(tuple(v[offset: offset + size] for v in vecs))
            offset += size
        return result


class GGNAccumulator():
    """
    Accumulates GGN matrices over batches and returns their running mean
    (without storing the full batch history).

    If param_sizes is provided, only the block-diagonal part of the GGN is
    kept: each block corresponds to one layer, extracted from the diagonal
    of the full (n_params, n_params) GGN passed to update().
    """

    def __init__(self, param_sizes=None):
        """
        Args:
            param_sizes: if provided, only diagonal blocks of the GGN are
                accumulated (one per layer), rather than the full matrix.
        """
        self.param_sizes = param_sizes
        self.reset()
        print(f"Storing GGN as the preconditioner, with block structure: {self.param_sizes is not None}")

    def reset(self):
        self._n    = 0
        self._mean = None   # either one (n_params, n_params) tensor,
                             # or a list of (block_size, block_size) tensors
        self.ggn   = None

    def update(self, ggn: torch.Tensor) -> None:
        """Add one GGN sample (n_params, n_params full matrix)."""
        self._n += 1

        if self.param_sizes is not None:
            blocks = self._extract_blocks(ggn)
            if self._mean is None:
                self._mean = [b.clone() for b in blocks]
            else:
                for mean_block, block in zip(self._mean, blocks):
                    mean_block += (block - mean_block) / self._n
        else:
            if self._mean is None:
                self._mean = ggn.clone()
            else:
                self._mean += (ggn - self._mean) / self._n

    def finalise(self) -> torch.Tensor | list[torch.Tensor]:
        """
        Return the accumulated running-mean GGN.

        Returns:
            Full mode:  a single (n_params, n_params) GGN tensor.
            Block mode: a list of per-layer (block_size, block_size) tensors.
        """
        if self._n < 1:
            raise RuntimeError("Need at least 1 GGN sample to compute the mean GGN.")

        self.ggn = self._mean
        ggn = self.ggn

        self.reset()
        return ggn

    def _extract_blocks(self, ggn: torch.Tensor) -> list[torch.Tensor]:
        """Extract per-layer diagonal blocks from the full GGN matrix."""
        blocks, offset = [], 0
        for size in self.param_sizes:
            blocks.append(ggn[offset: offset + size, offset: offset + size])
            offset += size
        return blocks


class Preconditioner():
    """
    Accumulates noisy gradients, estimates their covariance, and applies
    the inverse covariance to a gradient vector.

    Inverse methods:
        PINV   - pseudoinverse via SVD, handles rank-deficient matrices
        CHOLESKY - fast for well-conditioned matrices, requires damping
        EIGEN  - eigendecomposition, best for low-rank gradient covariances,
                 allows spectrum inspection and aggressive truncation
        CG     - conjugate gradient, best for large networks, no factorisation

    The factorisation is computed once in finalise() and cached, so each
    apply() is a cheap matrix-vector operation.
    """

    class InverseMethod(Enum):
        PINV     = "pinv"
        CHOLESKY = "cholesky"  # sensitive to the selection of damping scalar
        EIGEN    = "eigen"
        CG       = "cg"  # may diverge for nns

        @classmethod
        def from_str(cls, value: str) -> "Preconditioner.InverseMethod":
            try:
                return cls(value.lower())
            except ValueError:
                valid = [m.value for m in cls]
                raise ValueError(
                    f"Unknown inverse method: '{value}'. Choose from: {valid}")

    def __init__(self,
                 inverse_method: "Preconditioner.InverseMethod | str" = None,
                 damping: float = 1e-2,
                 rcond: float = 1e-3,
                 cg_max_iter: int = 50,
                 cg_tol: float = 1e-5,
                 param_sizes: list[int] | None = None,
                 noise_mult: float = 1.0,
                 use_ggn: bool = False
                 ):
        """
        Args:
            inverse_method: how to invert the covariance matrix. Accepts either
                            an InverseMethod enum value or a plain string (e.g.
                            from a config file). Defaults to EIGEN.
            damping:        λ for CHOLESKY mode: solves (C + λI).
            rcond:          threshold for small eigenvalues/singular values in
                            PINV and EIGEN modes.
            cg_max_iter:    maximum CG iterations per apply() call.
            cg_tol:         residual tolerance for early CG termination.
            param_sizes:    if provided, applies the preconditioner block-diagonally
                            per layer. Obtain via [p.numel() for p in model.parameters()].
            noise_mult:    std for the noise added to the gradients before covariance estimation.
            use_ggn:       a bool flag indicating that GNN martix is used instead of the approximation
        """
        if isinstance(inverse_method, str):
            inverse_method = Preconditioner.InverseMethod.from_str(inverse_method)
        self.inverse_method = inverse_method or Preconditioner.InverseMethod.EIGEN
        self.damping        = damping
        self.rcond          = rcond
        self.cg_max_iter    = cg_max_iter
        self.cg_tol         = cg_tol
        self.param_sizes    = param_sizes
        self.noise_mult     = noise_mult
        self.alpha_star     = None  # for adaptive step size
        self.tau_adapter    = BotevTauAdapter()  # for adaptive alpha damping

        self.use_ggn        = use_ggn
        if self.use_ggn:
            self._cov_estimator = GGNAccumulator(param_sizes)
        else:
            self._cov_estimator = GradientCovarianceEstimator(param_sizes)
        self._cov_recalc    = False
        # cached factorisations — type depends on inverse_method:
        #   PINV:     dense (n, n) inverse tensor, or list of tensors for block
        #   CHOLESKY: Cholesky factor L, or list of L for block
        #   EIGEN:    (eigvecs, eigvals_inv) tuple, or list of tuples for block
        #   CG:       raw covariance tensor (no factorisation), or list for block
        self._factorisation = None

    def update(self, grad: torch.Tensor) -> None:
        """Add one noisy gradient sample to the covariance estimate."""
        self._cov_estimator.update(grad)
        self._cov_recalc = True

    def finalise(self, grad=None) -> None:
        """Compute covariance from accumulated gradients and cache the factorisation.
        Skipped if no new gradients have been added since the last call.
        """
        if not self._cov_recalc:
            return

        cov = self._cov_estimator.finalise()
        if self.noise_mult != 1.0:
            if isinstance(cov, torch.Tensor):
                cov = cov / self.noise_mult ** 2
            else:
                cov = [block / self.noise_mult ** 2 for block in cov]

        self._factorisation = self._factorise_cov(cov)
        self._cov_recalc = False
        self.alpha_star = self.get_adaptive_alpha(grad, cov)

    def get_adaptive_alpha(self, grad, cov) -> float:
        """Calculate the optimal step size for the preconditioned gradient.
        alpha* = gᵀd / dᵀGd, where g is the raw gradient and d = C^{-1}g is the preconditioned gradient.
        """
        if self._factorisation is None or grad is None:
            return 1  # default small step size if preconditioner is not ready

        precond_grad = self.apply(grad)
        num = torch.dot(grad, precond_grad)
        if num <= 0:
            return 1  # fallback to a safe step size if non-positive curvature

        # calculate Gd product
        if isinstance(cov, torch.Tensor):
            Gd = cov @ precond_grad
        else:
            out, offset = [], 0
            for block_size, block_cov in zip(self.param_sizes, cov):
                grad_slice = precond_grad[offset: offset + block_size]
                out.append(block_cov @ grad_slice)
                offset += block_size
            Gd = torch.cat(out)

        dGd = torch.dot(precond_grad, Gd)  # dᵀGd > 0
        tau = self.tau_adapter.tau
        denom = dGd + tau * torch.dot(precond_grad, precond_grad)
        # print(f"Dtau is {self.tau:.4e}")
        # print(f"Adaptive alpha calculation: num={num:.4e}, denom={denom:.4e}, raw alpha={num / (denom + 1e-8):.4e}")

        alpha = num / (denom + 1e-8)
        alpha = alpha.clamp(1e-6, 5.0).item()

        # store for tau update
        self.tau_adapter.record_alpha(alpha, num)

        return alpha

    def apply(self, grad: torch.Tensor) -> torch.Tensor:
        """Return C^{-1} @ grad using the cached factorisation.
        Falls back to identity if finalise() has not been called yet.
        """
        if self._factorisation is None:
            return grad

        if isinstance(self._factorisation, list):
            # block-diagonal: apply each block to its gradient slice
            out, offset = [], 0
            for block_size, block_fact in zip(self.param_sizes, self._factorisation):
                grad_slice = grad[offset: offset + block_size]
                out.append(self._apply_factorisation(block_fact, grad_slice))
                offset += block_size
            return torch.cat(out)
        else:
            return self._apply_factorisation(self._factorisation, grad)

    # property for external inspection (e.g. logging eigenvalues)
    @property
    def eigenvalues(self) -> torch.Tensor | list[torch.Tensor] | None:
        """Returns eigenvalues if EIGEN mode, else None."""
        if self.inverse_method != Preconditioner.InverseMethod.EIGEN:
            return None
        if self._factorisation is None:
            return None
        if isinstance(self._factorisation, list):
            return [1.0 / eig_inv[eig_inv > 0] for _, eig_inv in self._factorisation]
        _, eigvals_inv = self._factorisation
        return 1.0 / eigvals_inv[eigvals_inv > 0]

    def _factorise_cov(self,
                       cov: torch.Tensor | list[torch.Tensor]
                       ) -> object:
        if isinstance(cov, list):
            return [self._factorise(block) for block in cov]
        return self._factorise(cov)

    def _factorise(self, matrix: torch.Tensor) -> object:
        method = self.inverse_method
        if method == Preconditioner.InverseMethod.PINV:
            return torch.linalg.pinv(matrix, rcond=self.rcond)

        elif method == Preconditioner.InverseMethod.CHOLESKY:
            n = matrix.shape[0]
            regularised = matrix + self.damping * torch.eye(n, device=matrix.device,
                                                            dtype=matrix.dtype)
            return torch.linalg.cholesky(regularised)   # store L, not L^{-1}

        elif method == Preconditioner.InverseMethod.EIGEN:
            # eigh exploits symmetry — faster than eig for covariance matrices
            eigvals, eigvecs = torch.linalg.eigh(matrix)
            eigvals_inv = torch.where(
                eigvals > self.rcond,
                1.0 / eigvals,
                torch.zeros_like(eigvals)
            )
            return (eigvecs, eigvals_inv)

        elif method == Preconditioner.InverseMethod.CG:
            return matrix   # no factorisation — CG works directly with C

        raise ValueError(f"Unknown inverse method: {self.inverse_method}")

    def _apply_factorisation(self, factorisation: object,
                             grad: torch.Tensor) -> torch.Tensor:
        method = self.inverse_method
        if method == Preconditioner.InverseMethod.PINV:
            return factorisation @ grad

        elif method == Preconditioner.InverseMethod.CHOLESKY:
            # two triangular solves: L L^T x = grad
            return torch.cholesky_solve(
                grad.unsqueeze(1), factorisation
            ).squeeze(1)

        elif method == Preconditioner.InverseMethod.EIGEN:
            eigvecs, eigvals_inv = factorisation
            # eigvecs @ diag(eigvals_inv) @ eigvecs.T @ grad
            return eigvecs @ (eigvals_inv * (eigvecs.T @ grad))

        elif method == Preconditioner.InverseMethod.CG:
            return self._apply_cg(factorisation, grad)

        raise ValueError(f"Unknown inverse method: {self.inverse_method}")

    def _apply_cg(self, cov: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Solve cov @ x = grad iteratively via conjugate gradient."""
        x     = torch.zeros_like(grad)
        r     = grad.clone()
        p     = r.clone()
        r_dot = torch.dot(r, r)

        for _ in range(self.cg_max_iter):
            Cp        = cov @ p
            alpha     = r_dot / torch.dot(p, Cp)
            x        += alpha * p
            r        -= alpha * Cp
            r_dot_new = torch.dot(r, r)
            if r_dot_new < self.cg_tol:
                break
            p         = r + (r_dot_new / r_dot) * p
            r_dot     = r_dot_new

        return x


class BotevTauAdapter:
    """
    Botev Appendix B.3.1 - LM tau adaptation, once per epoch.
    Matches the cov_pc update frequency.
    See https://arxiv.org/pdf/1706.03662
    """
    def __init__(self, tau_init=1., omega_tau=0.9):
        self.tau = tau_init
        self._omega = omega_tau
        self._num = None  # gᵀd stored from alpha computation
        self._alpha = None  # alpha* stored from alpha computation
        self._f_prev = None  # epoch start loss

    def record_alpha(self, alpha, num):
        """
        Called by get_adaptive_alpha to store num and alpha for the tau update.
        """
        self._alpha = alpha
        self._num = num.item() if torch.is_tensor(num) else num

    def record_f_prev(self, f_prev):
        """
        Called by to record the previous value of loss
        """
        self._f_prev = f_prev

    def update(self, f_new):
        """
        Call once at the end of each epoch with the aggregated
        train losses from the previous and current epoch.

        f_prev: mean train loss from epoch N-1
        f_new:  mean train loss from epoch N
        """
        if self._alpha is None or self._num is None or self._f_prev is None:
            self._f_prev = f_new
            return  # not enough info yet (epoch 1)

        predicted_decrease = -self._alpha * self._num
        if abs(predicted_decrease) < 1e-12:
            return

        rho = (f_new - self._f_prev) / predicted_decrease

        if rho > 0.75:
            self.tau *= self._omega  # good step, reduce damping
        else:
            self.tau /= self._omega  # poor step, more damping

        self.tau = float(max(1e-4, min(self.tau, 10.0)))
        self._f_prev = f_new
