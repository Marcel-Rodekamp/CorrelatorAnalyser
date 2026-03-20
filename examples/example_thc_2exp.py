"""
example_thc_2exp.py
======================
Minimal example: two-exponential correlator fit via the thc backend.

Layout
------
  1. Generate synthetic mock data (two-exponential signal + noise) with an
     exponentially decaying inter-timeslice covariance.
  2. Resample the raw data using Data.
  3. Define the model and log-normal priors, then call fit(...).
  4. Plot:
       • Top row  – data (with error bars) + fit curve.
       • 2×2 grid – parameter histograms over bootstrap resamples,
                    mean (solid), ±1 σ_bst (dashed), and ground truth (dotted).
"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from correlatoranalyser import Data
from correlatoranalyser.fit import fit
from correlatoranalyser.prior import Prior


# =============================================================================
# Ground-truth parameters & data-generation settings
# =============================================================================

A0_TRUE   = 1.5
E0_TRUE   = 0.3
A1_TRUE   = 0.75
E1_TRUE   = 0.8       
NOISE     = 0.02
NRAW      = 1500
NBST      = 500
T_MIN, T_MAX = 0, 24

# Derived quantities
t = np.arange(T_MIN, T_MAX + 1, dtype=float)   # timeslices  t = 1 … 24
Nt = len(t)


# =============================================================================
# 1. Create correlated mock data
# =============================================================================

rng = np.random.default_rng(seed=42)

# Noiseless two-exponential signal at each timeslice
signal = A0_TRUE * np.exp(-E0_TRUE * t) + A1_TRUE * np.exp(-E1_TRUE * t)

# Covariance matrix: diagonal noise² * exp(-|i-j| / xi)  (xi = 4 timeslices)
xi = 4.0
i_idx, j_idx = np.meshgrid(np.arange(Nt), np.arange(Nt), indexing="ij")
cov_raw = NOISE**2 * np.exp(-np.abs(i_idx - j_idx) / xi)

# Draw NRAW raw "configurations" from this multivariate Gaussian
raw = rng.multivariate_normal(mean=signal, cov=cov_raw, size=NRAW)

# Resample (Nresample = NBST)
ordinate = Data(resample_type="bst", data=raw, Nresample=NBST)


# =============================================================================
# 2. Run the fit
# =============================================================================

fit_result = fit(
    backend                      = "thc",
    ordinate                     = ordinate,
    central_value_fit            = True,
    central_value_fit_correlated = False,
    resample_fit                 = True,
    resample_fit_correlated      = False,  
    delta_t                      = 1,
    truncation_dimension         = None,
    truncation_method            = "gap",
    symmetric_correlator         = False,
    epsilon_real                 = 1e-8,
    epsilon_imag                 = 1e-8,
    t0                           = 0,
)

print(fit_result)


# =============================================================================
# 3. Diagnostic plots
# =============================================================================

# --- ground-truth values keyed by parameter name ---
TRUE_VALS = {
    "A0": A0_TRUE,
    "E0": E0_TRUE,
    "A1": A1_TRUE,
    "E1": E1_TRUE,
}

PARAM_LABELS = {
    "A0": r"$A_0$",
    "E0": r"$E_0$",
    "A1": r"$A_1$",
    "E1": r"$E_1$",
}

param_keys = list(TRUE_VALS.keys())   # ["A0", "E0", "A1", "E1"]

# ── figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 10))
fig.suptitle("Two-Exponential Correlator Fit  (thc backend)", fontsize=14, y=0.98)

# outer: 2 rows (top = data panel; bottom = 2×2 parameter histograms)
outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.42, height_ratios=[1.2, 1.8])

# ── top panel: data + fit ──────────────────────────────────────────────────
ax_top = fig.add_subplot(outer[0])

# data: log scale shows the two exponentials clearly
ax_top.set_yscale("log")
ax_top.errorbar(
    t, ordinate.mean, yerr=ordinate.serr,
    fmt="o", ms=5, color="#2166ac", elinewidth=1.2, capsize=3,
    label="Mock data", zorder=3,
)

# fit curve (dense abscissa for a smooth line)
t_dense = np.linspace(T_MIN, T_MAX, 300)
cv_params = {k: v.mean for k, v in fit_result.params.items()}
fit_curve = fit_result.fcn(t_dense, cv_params)
ax_top.plot(t_dense, fit_curve, color="#d6604d", lw=2, label="Fit (CV)", zorder=2)

# ground-truth noiseless signal
ax_top.plot(t_dense,
            A0_TRUE * np.exp(-E0_TRUE * t_dense) + A1_TRUE * np.exp(-E1_TRUE * t_dense),
            color="k", lw=1, ls="--", label="Ground truth", zorder=1)

ax_top.set_xlabel(r"$t$", fontsize=12)
ax_top.set_ylabel(r"$C(t)$", fontsize=12)
ax_top.set_xlim(T_MIN - 0.5, T_MAX + 0.5)
ax_top.legend(fontsize=10, framealpha=0.8)

# χ²/dof annotation
chi2_per_dof = fit_result.chi2.mean / fit_result.dof
p_val        = fit_result.p_value.mean
ax_top.set_title(
    rf"$\chi^2/\mathrm{{dof}} = {chi2_per_dof:.2f}$,  "
    rf"$p = {p_val:.3f}$,  "
    rf"$\mathrm{{dof}} = {fit_result.dof}$",
    fontsize=11,
)

# ── bottom: 2×2 parameter histograms ──────────────────────────────────────
inner = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=outer[1], hspace=0.55, wspace=0.35
)

HIST_COLOR  = "#4393c3"
MEAN_COLOR  = "#d6604d"
TRUTH_COLOR = "k"

for idx, key in enumerate(param_keys):
    ax = fig.add_subplot(inner[idx // 2, idx % 2])
 
    param   = fit_result.params[key]
    samples = param.rspl            # shape (NBST,)
    mu      = param.mean            # central value (CV fit)
    sigma   = param.serr            # bootstrap standard error
 
    # histogram of bootstrap resamples (density=True → integrates to 1)
    ax.hist(
        samples, bins=35, density=True,
        color=HIST_COLOR, alpha=0.75, edgecolor="white", linewidth=0.4,
    )
 
    # Gaussian N(mu, sigma) evaluated over a dense x-grid spanning ±4σ
    x_gauss = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    gauss   = np.exp(-0.5 * ((x_gauss - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(x_gauss, gauss,
            color=MEAN_COLOR, lw=2.0,
            label=rf"${PARAM_LABELS[key][1:-1]} = {param.gvar()}$")
 
    # Ground-truth Gaussian N(truth, sigma) — same width, shifted centre
    truth   = TRUE_VALS[key]
    ax.axvline( truth, color = TRUTH_COLOR, ls = ':', label=rf"${PARAM_LABELS[key][1:-1]}^\mathrm{{true}} = {truth:.4f}$")
 
    ax.set_xlabel(PARAM_LABELS[key], fontsize=11)
    ax.set_ylabel("density", fontsize=9)
    ax.set_title(PARAM_LABELS[key], fontsize=12)
    ax.legend(fontsize=7.5, framealpha=0.7, loc="upper right")

plt.savefig("plot_thc_2exp.pdf", bbox_inches="tight")