"""
Out-of-Distribution (OOD) Regime Shift & Tail-Risk Telemetry.

Validates time-series model stability under non-ergodic structural breaks:
1. Strict Chronological In-Sample (IS) vs. Out-of-Sample (OOS) Partitioning.
2. Synthetic Merton Jump-Diffusion Perturbations for Poisson Tail-Risk Testing.
3. Maximum Drawdown & Error Degradation Ratio Analysis.
"""

from typing import Dict, Tuple
import numpy as np


class RegimeShiftValidator:
    def __init__(self, is_ratio: float = 0.7):
        """
        Args:
            is_ratio: In-Sample proportion (default 0.7 for 70:30 chronological split).
        """
        if not (0.0 < is_ratio < 1.0):
            raise ValueError("In-sample ratio must be strictly between 0 and 1.")
        self.is_ratio = is_ratio

    def chronological_split(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits temporal series into In-Sample and Out-of-Sample without future lookahead.
        """
        T = series.shape[-1]
        split_idx = int(T * self.is_ratio)
        if series.ndim == 1:
            return series[:split_idx], series[split_idx:]
        return series[..., :split_idx], series[..., split_idx:]

    def inject_poisson_jump_diffusion(
        self,
        series: np.ndarray,
        lambda_shock: float = 0.05,
        shock_magnitude: float = 3.0,
        seed: int = 42
    ) -> np.ndarray:
        """
        Simulates Merton Jump-Diffusion structural breaks:
        dS_t = mu dt + sigma dW_t + J_t dN_t(lambda)

        Args:
            series: Ground truth series of shape (..., T).
            lambda_shock: Poisson intensity for arrival of structural regime shifts.
            shock_magnitude: Standard deviation scale of jump shock.
        """
        rng = np.random.default_rng(seed)
        perturbed = np.copy(series)
        T = series.shape[-1]

        # Poisson jump occurrences
        jump_occurrences = rng.poisson(lam=lambda_shock, size=T) > 0
        jump_sizes = rng.normal(loc=0.0, scale=shock_magnitude * np.std(series), size=T)

        cum_jump = np.cumsum(jump_occurrences * jump_sizes)
        perturbed += cum_jump
        return perturbed

    def compute_stress_telemetry(
        self,
        y_true: np.ndarray,
        y_pred_baseline: np.ndarray,
        y_pred_model: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluates structural break resilience and error degradation ratio.
        """
        rmse_base = np.sqrt(np.mean((y_true - y_pred_baseline) ** 2))
        rmse_model = np.sqrt(np.mean((y_true - y_pred_model) ** 2))
        degradation_ratio = (rmse_model / rmse_base) if rmse_base > 0 else 1.0

        # Max error drawdown
        abs_err = np.abs(y_true - y_pred_model)
        max_drawdown = float(np.max(abs_err))

        return {
            "rmse_baseline": float(rmse_base),
            "rmse_model": float(rmse_model),
            "degradation_ratio": float(degradation_ratio),
            "max_error_drawdown": max_drawdown,
            "relative_improvement_pct": float((rmse_base - rmse_model) / max(rmse_base, 1e-8) * 100.0)
        }
