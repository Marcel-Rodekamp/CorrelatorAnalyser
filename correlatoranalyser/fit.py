from .fitResult import FitResult

def fit(
    backend: str = "iminuit",
    **kwargs
) -> FitResult:

    if backend == "iminuit":
        if "linear_params" in kwargs:
            from .fit_iminuit_variable_projection import fit_iminuit
            return fit_iminuit( **kwargs )
        else:
            from .fit_iminuit import fit_iminuit
            return fit_iminuit( **kwargs )
    
    elif backend == "lsqfit":
        from .fit_lsqfit import fit_lsqfit
        return fit_lsqfit ( **kwargs )
    
    elif backend == "linear regression":
        from .fit_linear_regression import linear_regression
        return linear_regression( **kwargs )
    else:
        raise ValueError(f"Couldn't identify backkend {backend}. Must be one of ['iminuit', 'lsqfit', 'linear regression']")
    