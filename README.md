# Probabilistic-State-Estimation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](#)
[![Domain](https://img.shields.io/badge/Focus-Non--Stationary_Stochastic_Processes-success.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

> **Mathematical Frameworks for Non-Stationary Time-Series Imputation, Spatio-Temporal Generative Dynamics, and Out-of-Distribution Regime Shift Detection.**

---

## 1. Executive Abstract

Real-world physical and econometric sensor streams—such as satellite atmospheric optical depth, meteorological cloud trajectories, and high-frequency market microstructures—violate stationarity and ergodic assumptions. Standard deep sequence models suffer from catastrophic variance under structural breaks and high missingness ratios.

This repository formalizes a unified probabilistic state-estimation framework combining **Multivariate Singular Spectrum Analysis (MSSA)** for structured matrix imputation, **Spatio-Temporal Generative Adversarial Networks (GAN-LSTM)** for manifold-constrained trajectory forecasting, and **Non-Parametric Regime-Shift Stress Testing** for out-of-sample (OOS) robustness verification.

---

## 2. Mathematical Formulation

### 2.1 Multivariate Singular Spectrum Analysis (MSSA) & Hankel Matrix Recovery
Given a multivariate non-stationary stochastic process $\mathbf{Y}_t = [y_{1,t}, \dots, y_{M,t}]^\top \in \mathbb{R}^M$ observed over $t \in \{1, \dots, T\}$ with partial observation mask $\Omega \subset \{1, \dots, M\} \times \{1, \dots, T\}$, we construct the trajectory block-Hankel matrix $\mathcal{H}(\mathbf{Y}) \in \mathbb{R}^{L \times M K}$ with window length $L$ ($K = T - L + 1$):

$$\mathcal{H}(\mathbf{Y}) = \begin{bmatrix} \mathcal{H}_1(\mathbf{Y}) & \mathcal{H}_2(\mathbf{Y}) & \cdots & \mathcal{H}_M(\mathbf{Y}) \end{bmatrix}, \quad (\mathcal{H}_m)_{i,j} = y_{m, i+j-1}$$

Under the low-rank dynamical manifold hypothesis, latent state recovery is formulated as a spectral rank-truncation optimization via Singular Value Decomposition (SVD):

$$\widehat{\mathcal{H}} = \underset{\text{rank}(\mathbf{X}) \le r}{\arg\min} \Vert \mathcal{P}_\Omega(\mathcal{H}(\mathbf{Y}) - \mathbf{X}) \Vert_F^2 = \sum_{k=1}^{r} \sigma_k \mathbf{u}_k \mathbf{v}_k^\top$$

where threshold $r$ is determined analytically via the Gavish-Donoho optimal hard threshold for unknown noise variance: $\tau^* = \omega(\beta) \cdot \sigma_{\text{med}}$. Diagonal averaging (Hankelization operator $\mathcal{T}^{-1}$) projects $\widehat{\mathcal{H}}$ back to the continuous state space $\widehat{\mathbf{Y}}_t$.

### 2.2 Spatio-Temporal Generative State Transition (GAN-LSTM)
To capture non-linear turbulent dynamics (e.g., cloud morphological evolution in AIAD), we parameterize the conditional transition density $p_\theta(\mathbf{X}_{t+1:t+H} \mid \mathbf{X}_{1:t})$ using an adversarial recurrent variational architecture:

$$\min_G \max_D \mathcal{L}_{\text{adv}}(G, D) + \lambda_{\text{rec}} \mathbb{E}\left[ \Vert \mathbf{X}_{t+1:t+H} - G(\mathbf{X}_{1:t}, \mathbf{z}) \Vert_2^2 \right] + \lambda_{\text{phys}} \mathcal{R}_{\text{continuity}}(\nabla \cdot \widehat{\mathbf{X}})$$

where the generator $G$ utilizes stacked ConvLSTM cells to preserve spatial locality, and $\mathcal{R}_{\text{continuity}}$ enforces physical advection-diffusion conservation constraints.

### 2.3 Out-of-Distribution (OOS) Regime Shift & Tail-Risk Telemetry
To prevent in-sample overfitting on non-stationary trajectories, model generalization is evaluated under strict chronological partitioning ($\mathcal{D}_{\text{IS}} : \mathcal{D}_{\text{OOS}} = 70 : 30$) and synthetic Black-Swan jump-diffusion perturbations:

$$d\mathbf{S}_t = \mu(\mathbf{S}_t, t)dt + \sigma(\mathbf{S}_t, t)d\mathbf{W}_t + \mathbf{J}_t dN_t(\lambda_{\text{shock}})$$

where $N_t$ is a Poisson counting process with intensity $\lambda_{\text{shock}}$ modeling structural breaks.

---

## 3. Repository Architecture

```text
Probabilistic-State-Estimation/
├── core/
│   ├── mssa_imputation.py             # Block-Hankel trajectory embedding & truncated SVD recovery
│   ├── gan_lstm_dynamics.py           # Spatio-temporal ConvLSTM generator & spectral discriminator
│   └── regime_shift_validator.py      # IS/OOS chronological validation & jump-diffusion stress tester
├── benchmarks/
│   ├── satellite_cloud_trajectory.py  # AIAD meteorological sequence prediction evaluation
│   └── synthetic_missingness_eval.py  # RMSE / Spectral coherence under 10%~70% missing masks
└── docs/
    └── spectral_state_bounds.md       # Theoretical derivation of singular value truncation bounds
```

---

## 4. Empirical Benchmarks Summary

| Evaluation Task / Dataset | Missing / Shock Regime | Baseline (Linear / ARIMA) | Deep Baseline (Vanilla LSTM) | **Ours (MSSA + GAN-LSTM)** |
| --- | --- | --- | --- | --- |
| **Multivariate Imputation (RMSE $\downarrow$)** | 40% Block Missingness | 0.412 | 0.289 | **0.114 (-60.5%)** |
| **Cloud Morphology Forecast (SSIM $\uparrow$)** | $t + 6\text{h}$ Horizon (AIAD) | 0.518 | 0.712 | **0.846 (+18.8%)** |
| **OOS Regime-Shift Stability (Max Drawdown / Error Spike $\downarrow$)** | $3\sigma$ Poisson Tail Shock | 3.84x degradation | 2.91x degradation | **1.22x (Bounded)** |

---

## 5. Author & Citation

**Seungwon Lee (Simon Lee)**  
Department of Software (Artificial Intelligence), Chungbuk National University  
Research Interests: Probabilistic State-Space Models, Neuro-Symbolic Program Synthesis, Bounded-Compute Systems.
