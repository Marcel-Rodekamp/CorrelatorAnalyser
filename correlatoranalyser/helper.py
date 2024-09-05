import numpy as np

import gvar as gv


def meff(
    Corr_est: np.ndarray, Corr_err: np.ndarray, Corr_bst: np.ndarray, delta: float = 1
):
    r""" """
    if Corr_est is None and Corr_bst is None:
        raise RuntimeError(f"Either provide Corr_est and Corr_err or Corr_bst")

    if Corr_err is None and Corr_bst is None:
        raise RuntimeError(f"Either provide Corr_est and Corr_err or Corr_bst")

    if Corr_bst is None:
        Corr_gvar = gv.gvar(Corr_est, Corr_err)

        # No bootstrap time axis assumed in zeroth axis
        meff_gvar = (
            gv.log(np.roll(Corr_gvar, -1, axis=0))
            - gv.log(np.roll(Corr_gvar, 1, axis=0))
        ) / (2 * delta)

        return None, gv.mean(meff_gvar), gv.sdev(meff_gvar)

    else:
        # No bootstrap time axis assumed in first axis
        meff_bst = (
            np.log(np.roll(Corr_bst, -1, axis=1)) - np.log(np.roll(Corr_bst, 1, axis=1))
        ) / (2 * delta)

        return meff_bst, np.nanmean(meff_bst, axis=0), np.nanstd(meff_bst, axis=0)
