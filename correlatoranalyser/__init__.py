# Class that organizes resampling
# in particular bootstrap and jackknife
from .data import Data 

from .prior import Prior

# Class that stores/organizes results of fits
# result of fit(...)
from .fitResult import FitResult

# Collection of FitResults. It allows to simplify model averaging etc.
from .fitState import FitState

# A method that with provided data performs fit and return a FitResult
from .fit import fit

# A explicit linear regression
# After v1.0 this is implemented in the fit(backend='linear regression') method 
#from .fit_linearRegression import linear_regression
