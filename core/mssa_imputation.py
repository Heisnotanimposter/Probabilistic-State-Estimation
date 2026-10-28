"""
Multivariate Singular Spectrum Analysis (MSSA) for Non-Stationary Time-Series Imputation.

Mathematical Formulation:
Given multivariate trajectories Y in R^{M x T} with missing mask Omega:
1. Construct block-Hankel trajectory matrix H(Y) in R^{L x M*K}.
2. Solve low-rank SVD truncation: \hat{H} = argmin_{rank(X) <= r} ||P_Omega(H(Y) - X)||_F^2.
3. Compute Gavish-Donoho optimal hard threshold tau* for singular values.
4. Invert via diagonal averaging (Hankelization operator) back to continuous series.
"""

from typing import Optional, Tuple
import numpy as np


class MultivariateSSA:
    def __init__(self, window_length: int, rank_cutoff: Optional[int] = None):
        """
        Args:
            window_length (L): Embedding window length (must satisfy 2 <= L <= T // 2).
            rank_cutoff (r): Explicit truncation rank. If None, Gavish-Donoho threshold is used.
        """
        self.L = window_length
        self.r = rank_cutoff
        self.singular_values_ = None
        self.explained_variance_ratio_ = None

    def _build_block_hankel(self, Y: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
        """
        Constructs block-Hankel matrix H = [H_1, H_2, ..., H_M] from Y of shape (M, T).
        """
        M, T = Y.shape
        L = self.L
        K = T - L + 1
        if K <= 0:
            raise ValueError(f"Window length L={L} exceeds series length T={T}.")

        H_blocks = []
        for m in range(M):
            series = Y[m, :]
            # Vectorized Hankel construction
            sub_H = np.empty((L, K), dtype=Y.dtype)
            for i in range(L):
                sub_H[i, :] = series[i : i + K]
            H_blocks.append(sub_H)

        H = np.hstack(H_blocks)
        return H, M, L, K

    def _invert_diagonal_averaging(self, H: np.ndarray, M: int, L: int, K: int, T: int) -> np.ndarray:
        """
        Projects truncated block-Hankel matrix back to time series Y of shape (M, T).
        """
        Y_recovered = np.zeros((M, T), dtype=H.dtype)
        for m in range(M):
            sub_H = H[:, m * K : (m + 1) * K]
            series = np.zeros(T, dtype=H.dtype)
            counts = np.zeros(T, dtype=int)
            for i in range(L):
                for j in range(K):
                    series[i + j] += sub_H[i, j]
                    counts[i + j] += 1
            Y_recovered[m, :] = series / np.maximum(counts, 1)
        return Y_recovered

    def _gavish_donoho_rank(self, s: np.ndarray, m: int, n: int) -> int:
        """
        Optimal hard threshold for singular values in unknown white noise (Gavish & Donoho, 2014).
        """
        beta = min(m, n) / max(m, n)
        # Numerical approximation for omega(beta)
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        s_med = np.median(s)
        tau = omega * s_med
        rank = np.sum(s >= tau)
        return max(1, int(rank))

    def fit_transform(self, Y: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Imputes missing values and decomposes non-stationary multivariate trajectories.

        Args:
            Y: Input array of shape (M, T) where M is feature channels and T is time steps.
            mask: Boolean array of shape (M, T), True indicates observed, False indicates missing.

        Returns:
            Y_imputed: Reconstructed matrix of shape (M, T).
        """
        Y_clean = np.copy(Y)
        if mask is not None:
            # Initial mean imputation across observed values per channel
            for m in range(Y.shape[0]):
                obs = Y[m, mask[m, :]]
                fill_val = np.mean(obs) if len(obs) > 0 else 0.0
                Y_clean[m, ~mask[m, :]] = fill_val

        H, M, L, K = self._build_block_hankel(Y_clean)
        U, s, Vt = np.linalg.svd(H, full_matrices=False)
        self.singular_values_ = s
        total_var = np.sum(s**2)
        self.explained_variance_ratio_ = (s**2) / total_var if total_var > 0 else np.zeros_like(s)

        # Determine truncation rank
        target_r = self.r if self.r is not None else self._gavish_donoho_rank(s, H.shape[0], H.shape[1])
        target_r = min(target_r, len(s))

        # Truncated reconstruction
        H_hat = (U[:, :target_r] * s[:target_r]) @ Vt[:target_r, :]
        Y_reconstructed = self._invert_diagonal_averaging(H_hat, M, L, K, Y.shape[1])

        # Preserve exact observed points if mask provided
        if mask is not None:
            Y_reconstructed[mask] = Y[mask]

        return Y_reconstructed
