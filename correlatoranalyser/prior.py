from __future__ import annotations

import numpy as np
import h5py as h5
from typing import Self

# Valid distribution names and their canonical form
_VALID_DISTS = {
    "normal"    : "normal",
    "log-normal": "log-normal",
    "log"       : "log-normal",
}


class Prior:
    """
    A simple Gaussian or log-normal prior for use in chi-squared fits.

    The prior contribution to chi-squared is

        normal:     ((theta - mean) / sdev)^2
        log-normal: ((log(theta) - mean) / sdev)^2

    Parameters
    ----------
    mean : float
        Central value of the prior.
        For log-normal this is the mean of log(theta), not of theta itself.
    sdev : float
        Standard deviation of the prior.
    dist : str
        Distribution family: 'normal', 'log-normal', or 'log'. (default: 'normal')
    """

    mean: float
    sdev: float
    dist: str

    def __init__(self, mean: float, sdev: float, dist: str = "normal") -> None:
        if dist not in _VALID_DISTS:
            raise ValueError(
                f"dist must be one of {list(_VALID_DISTS)}, got '{dist}'"
            )
        if sdev <= 0:
            raise ValueError(f"sdev must be positive, got {sdev}")

        self.mean = float(mean)
        self.sdev = float(sdev)
        self.dist = _VALID_DISTS[dist]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, theta: float) -> float:
        """Return the prior chi-squared contribution at parameter value theta."""
        if self.dist == "normal":
            return ((theta - self.mean) / self.sdev) ** 2
        else:
            if theta <= 0:
                # Return a large finite penalty.
                # np.inf is avoided because it can break finite-difference
                # gradient estimates at the boundary.
                return 1e30
            return ((np.log(theta) - self.mean) / self.sdev) ** 2

    # ------------------------------------------------------------------
    # Gradient/Jacobian
    # ------------------------------------------------------------------

    def grad(self, theta: float) -> float:
        """Return d(chi²_prior)/d(theta) — the gradient of the prior contribution."""
        if self.dist == "normal":
            return 2.0 * (theta - self.mean) / self.sdev ** 2
        else:
            if theta <= 0:
                # Consistent with __call__: d const / dx = 0
                return 0
            return 2.0 * (np.log(theta) - self.mean) / (self.sdev ** 2 * theta)

    # ------------------------------------------------------------------
    # gvar interoperability
    # ------------------------------------------------------------------

    def gvar(self):
        """Return a gvar.GVar representation of this prior."""
        import gvar as gv

        g = gv.gvar(self.mean, self.sdev)
        return gv.log(g) if self.dist == "log-normal" else g

    # ------------------------------------------------------------------
    # lsqfit interoperability
    # ------------------------------------------------------------------

    @staticmethod
    def import_from_lsqfit(prior: dict) -> dict[str, Prior]:
        """
        Convert a lsqfit prior dict (values are gvar.GVar) back to Prior objects.

        lsqfit encodes log-normal priors as ``"log(key)"`` keys.
        """
        import gvar as gv

        out: dict[str, Prior] = {}
        for key, value in prior.items():
            if key.startswith("log(") and key.endswith(")"):
                param_name = key[4:-1]
                out[param_name] = Prior(
                    mean = float(gv.mean(gv.exp(value))),
                    sdev = float(gv.sdev(value)),
                    dist = "log-normal",
                )
            else:
                out[key] = Prior(
                    mean = float(gv.mean(value)),
                    sdev = float(gv.sdev(value)),
                    dist = "normal",
                )
        return out

    # ------------------------------------------------------------------
    # HDF5 serialisation
    # ------------------------------------------------------------------

    def serialize(self, h5f: h5.Group, node: str | None = None) -> None:
        """Write this prior into an HDF5 group."""
        grp = h5f if node is None else h5f.require_group(node)
        grp.create_dataset("mean", data=self.mean)
        grp.create_dataset("sdev", data=self.sdev)
        grp.create_dataset("dist", data=self.dist)

    @staticmethod
    def deserialize(h5f: h5.Group, node: str | None = None) -> Prior:
        """Read a prior from an HDF5 group."""
        grp = h5f if node is None else h5f[node]

        dist = grp["dist"][()]
        if isinstance(dist, bytes):
            dist = dist.decode("utf-8")

        return Prior(
            mean = float(grp["mean"][()]),
            sdev = float(grp["sdev"][()]),
            dist = dist,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.dist == "normal":
            return f"N[μ={self.mean:g}, σ={self.sdev:g}]"
        else:
            return f"logN[μ={self.mean:g}, σ={self.sdev:g}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Prior):
            return NotImplemented
        return self.mean == other.mean and self.sdev == other.sdev and self.dist == other.dist
