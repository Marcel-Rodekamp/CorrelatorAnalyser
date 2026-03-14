"""
example_linear_regression.py
=============================
Minimal example: linear fit  y(x) = m·x + b  via the linear regression backend.

Layout
------
  1. Generate synthetic mock data (linear signal + noise) with an
     exponentially decaying inter-point covariance.
  2. Wrap the raw data in a bootstrap Data object.
  3. Call fit(...) — no model function or priors required for linear regression.
  4. Print the FitResult summary.
  5. Plot:
       • Top row  – data (with error bars) + fit line + ground truth.
       • 1×2 grid – parameter histograms over bootstrap resamples with a
                    fitted Gaussian on top and a dotted vertical line for the
                    ground truth.
"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from correlatoranalyser import Data
from correlatoranalyser.fit import fit


# =============================================================================
# Ground-truth parameters & data-generation settings
# =============================================================================

M_TRUE = 2.5       # slope
B_TRUE = 0.7       # intercept
NOISE  = 0.5       # per-point noise level
NRAW   = 1500
NBST   = 500
X_MIN, X_MAX = 1, 24


# =============================================================================
# 1. Create correlated mock data
# =============================================================================

rng = np.random.default_rng(seed=75132)

x  = np.arange(X_MIN, X_MAX + 1, dtype=float)   # x = 1 … 24
Nx = len(x)

# Noiseless linear signal
signal = M_TRUE * x + B_TRUE

# Covariance matrix: diagonal noise² * exp(-|i-j| / xi)
xi = 4.0
i_idx, j_idx = np.meshgrid(np.arange(Nx), np.arange(Nx), indexing="ij")
cov_raw = NOISE**2 * np.exp(-np.abs(i_idx - j_idx) / xi)

# Draw NRAW raw "configurations" from this multivariate Gaussian
raw = rng.multivariate_normal(mean=signal, cov=cov_raw, size=NRAW)

# Wrap into a bootstrap Data object (Nresample = NBST)
ordinate = Data(resample_type="bst", data=raw, Nresample=NBST)


# =============================================================================
# 2. Run the fit
#    Linear regression is closed-form — no model function or priors needed.
# =============================================================================

fit_result = fit(
    backend                      = "linear regression",
    abscissa                     = x,
    ordinate                     = ordinate,
    has_intercept                = True,
    parameter_names              = ("m", "b"),
    central_value_fit            = True,
    central_value_fit_correlated = True,
    resample_fit                 = True,
    resample_fit_correlated      = True,
)

print(fit_result)


# =============================================================================
# 3. Diagnostic plots
# =============================================================================

# --- ground-truth values keyed by parameter name ---
TRUE_VALS = {
    "m": M_TRUE,
    "b": B_TRUE,
}

PARAM_LABELS = {
    "m": r"$m$",
    "b": r"$b$",
}

param_keys = list(TRUE_VALS.keys())   # ["m", "b"]

# ── figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 9))
fig.suptitle("Linear Regression  (linear regression backend)", fontsize=14, y=0.98)

# outer: 2 rows (top = data panel; bottom = 1×2 parameter histograms)
outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45, height_ratios=[1.2, 1.4])

# ── top panel: data + fit ──────────────────────────────────────────────────
ax_top = fig.add_subplot(outer[0])

ax_top.errorbar(
    x, ordinate.mean, yerr=ordinate.serr,
    fmt="o", ms=5, color="#2166ac", elinewidth=1.2, capsize=3,
    label="Mock data", zorder=3,
)

# fit line (dense abscissa for a smooth line)
x_dense   = np.linspace(X_MIN, X_MAX, 300)
cv_params = {k: v.mean for k, v in fit_result.params.items()}
fit_line  = cv_params["m"] * x_dense + cv_params["b"]
ax_top.plot(x_dense, fit_line, color="#d6604d", lw=2, label="Fit (CV)", zorder=2)

# ground-truth noiseless signal
ax_top.plot(x_dense, M_TRUE * x_dense + B_TRUE,
            color="k", lw=1, ls="--", label="Ground truth", zorder=1)

ax_top.set_xlabel(r"$x$", fontsize=12)
ax_top.set_ylabel(r"$y(x)$", fontsize=12)
ax_top.set_xlim(X_MIN - 0.5, X_MAX + 0.5)
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

# ── bottom: 1×2 parameter histograms ──────────────────────────────────────
inner = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[1], hspace=0.55, wspace=0.35
)

HIST_COLOR  = "#4393c3"
MEAN_COLOR  = "#d6604d"
TRUTH_COLOR = "k"

for idx, key in enumerate(param_keys):
    ax = fig.add_subplot(inner[0, idx])

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

    # Ground-truth dotted vertical line
    truth = TRUE_VALS[key]
    ax.axvline(truth, color=TRUTH_COLOR, ls=":",
               label=rf"${PARAM_LABELS[key][1:-1]}^\mathrm{{true}} = {truth:.4f}$")

    ax.set_xlabel(PARAM_LABELS[key], fontsize=11)
    ax.set_ylabel("density", fontsize=9)
    ax.set_title(PARAM_LABELS[key], fontsize=12)
    ax.legend(fontsize=7.5, framealpha=0.7, loc="upper right")

plt.savefig("plot_linear_regression.pdf", bbox_inches="tight")